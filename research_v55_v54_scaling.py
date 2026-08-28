"""Screen conservative regime-specific scaling of the public-positive v54 correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v53_roster_stability import clustered_interval


ROOT = Path(__file__).resolve().parent
SCALES = (.75, .875, 1., 1.125, 1.25, 1.375, 1.5)


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    rows = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    target = rows["control_success"].to_numpy(float)
    futures = rows["game_type"].astype(str).eq("F").to_numpy()
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = np.clip(
            archive["blended"][archive["season"] == 2024].astype(float), .005, .995,
        )
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = np.clip(
            archive["blended"][archive["season"] == 2024].astype(float), .005, .995,
        )
    direction = logit(v54) - logit(v38)

    previous = raw.loc[raw["season"].eq(2023)]
    returning_pitcher = rows["pitcher_id"].isin(previous["pitcher_id"]).to_numpy()
    returning_batter = rows["batter_id"].isin(previous["batter_id"]).to_numpy()
    last_pitcher_team = previous.groupby("pitcher_id", observed=True).tail(1).set_index(
        "pitcher_id"
    )["pitcher_team_id"]
    last_batter_team = previous.groupby("batter_id", observed=True).tail(1).set_index(
        "batter_id"
    )["batter_team_id"]
    same_teams = (
        rows["pitcher_id"].map(last_pitcher_team).eq(rows["pitcher_team_id"])
        & rows["batter_id"].map(last_batter_team).eq(rows["batter_team_id"])
    ).to_numpy()
    cohorts = {**masks(len(rows)), "R": ~futures, "F": futures}
    cohorts.update({
        "returning_both": returning_pitcher & returning_batter,
        "roster_change": ~(returning_pitcher & returning_batter),
        "same_teams": same_teams,
        "player_or_team_change": ~same_teams,
    })
    base_scores = {
        name: float(bss(target[active], v54[active]))
        for name, active in cohorts.items() if int(active.sum()) >= 500
    }
    reports = []
    for r_scale in SCALES:
        for f_scale in SCALES:
            scale = np.where(futures, f_scale, r_scale)
            prediction = sigmoid(logit(v38) + scale * direction)
            gains = {
                name: float(bss(target[active], prediction[active])) - base_scores[name]
                for name, active in cohorts.items() if name in base_scores
            }
            team_gains = {}
            for team, index in rows.groupby("pitcher_team_id", observed=True).groups.items():
                active = np.zeros(len(rows), dtype=bool)
                active[np.asarray(list(index), dtype=int)] = True
                if int(active.sum()) >= 500:
                    team_gains[str(team)] = (
                        float(bss(target[active], prediction[active]))
                        - float(bss(target[active], v54[active]))
                    )
            reports.append({
                "r_scale": r_scale, "f_scale": f_scale, "gains": gains,
                "minimum_quarter_gain": min(gains[f"q{i}"] for i in range(1, 5)),
                "minimum_roster_gain": min(
                    gains[name] for name in (
                        "returning_both", "roster_change", "same_teams",
                        "player_or_team_change",
                    )
                ),
                "minimum_team_gain": min(team_gains.values()),
            })
    ranked = sorted(reports, key=lambda report: (
        report["minimum_quarter_gain"], report["minimum_roster_gain"],
        report["minimum_team_gain"], report["gains"]["all"],
    ), reverse=True)
    for report in ranked[:20]:
        scale = np.where(futures, report["f_scale"], report["r_scale"])
        prediction = sigmoid(logit(v38) + scale * direction)
        report["pitcher_clustered_bootstrap"] = clustered_interval(
            target, v54, prediction, rows["pitcher_id"].to_numpy(), repeats=2000,
        )
    safe = [report for report in ranked[:20] if (
        report["gains"]["all"] > 0.
        and report["minimum_quarter_gain"] > 0.
        and report["minimum_roster_gain"] > 0.
        and report["minimum_team_gain"] > 0.
        and report["pitcher_clustered_bootstrap"]["p05"] > 0.
    )]
    safe = sorted(safe, key=lambda report: report["gains"]["all"], reverse=True)
    output = {
        "anchor": {"version": "v54", "public_score": 1113},
        "safe": safe, "ranked": ranked,
        "row_independent": True, "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v55_v54_scaling.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe, "top": ranked[:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
