"""Audit agreement-gated amplification of the public-positive v54 correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v53_roster_stability import clustered_interval


ROOT = Path(__file__).resolve().parent
BASE_F_SCALE = 1.125
HIGH_SCALES = (1.25, 1.375, 1.5)
MIN_AGREEMENTS = (2, 3, 4)
MIN_ABS_DIRECTIONS = (0., .01, .02, .04)


def load(name, key):
    with np.load(ROOT / "research" / f"{name}.npz") as archive:
        return archive[key].astype(float)


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
    total_direction = logit(v54) - logit(v38)
    v55_scale = np.where(futures, BASE_F_SCALE, 1.)
    v55 = sigmoid(logit(v38) + v55_scale * total_direction)

    failure = load("v34_categorical_failure_lowcard_no_ids_hl2_2024", "new_failure")
    overlap = load("v45_overlap_hl2_2024", "overlap_probability")
    command = load("v43_multiclass_hl2_s1_2024", "prediction")
    recent_a = load("v48_regime_command_s3_2024", "multiclass")
    recent_b = load("v49_regime_multiclass_complexity_2024", "d6_i1000")
    joint = load("v52_pitch_command_joint_s3_2024", "history_no_team")
    corrected_failure = np.clip(failure + .45 * overlap, .005, .995)
    component_directions = np.column_stack([
        logit(command) - logit(v38),
        logit(corrected_failure) - logit(v38),
        logit(.5 * (recent_a + recent_b)) - logit(v38),
        logit(joint) - logit(v38),
    ])
    total_sign = np.sign(total_direction)[:, None]
    agreement = (np.sign(component_directions) == total_sign).sum(axis=1)

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
        name: float(bss(target[active], v55[active]))
        for name, active in cohorts.items() if int(active.sum()) >= 500
    }

    reports = []
    for high_scale in HIGH_SCALES:
        for minimum_agreement in MIN_AGREEMENTS:
            for minimum_direction in MIN_ABS_DIRECTIONS:
                selected = (
                    futures
                    & (agreement >= minimum_agreement)
                    & (np.abs(total_direction) >= minimum_direction)
                )
                scale = v55_scale.copy()
                scale[selected] = high_scale
                prediction = sigmoid(logit(v38) + scale * total_direction)
                gains = {
                    name: float(bss(target[active], prediction[active])) - base_scores[name]
                    for name, active in cohorts.items() if name in base_scores
                }
                selected_gain = None
                if int(selected.sum()) >= 500:
                    selected_gain = (
                        float(bss(target[selected], prediction[selected]))
                        - float(bss(target[selected], v55[selected]))
                    )
                team_gains = {}
                for team, index in rows.groupby("pitcher_team_id", observed=True).groups.items():
                    active = np.zeros(len(rows), dtype=bool)
                    active[np.asarray(list(index), dtype=int)] = True
                    if int((active & selected).sum()) >= 100:
                        team_gains[str(team)] = (
                            float(bss(target[active], prediction[active]))
                            - float(bss(target[active], v55[active]))
                        )
                reports.append({
                    "high_scale": high_scale,
                    "minimum_agreement": minimum_agreement,
                    "minimum_abs_direction": minimum_direction,
                    "selected_rows": int(selected.sum()),
                    "selected_gain": selected_gain,
                    "gains": gains,
                    "minimum_quarter_gain": min(gains[f"q{i}"] for i in range(1, 5)),
                    "minimum_roster_gain": min(
                        gains[name] for name in (
                            "returning_both", "roster_change", "same_teams",
                            "player_or_team_change",
                        )
                    ),
                    "minimum_affected_team_gain": (
                        min(team_gains.values()) if team_gains else None
                    ),
                })
    ranked = sorted(reports, key=lambda report: (
        report["minimum_quarter_gain"], report["minimum_roster_gain"],
        report["gains"]["all"],
    ), reverse=True)
    for report in ranked[:20]:
        selected = (
            futures
            & (agreement >= report["minimum_agreement"])
            & (np.abs(total_direction) >= report["minimum_abs_direction"])
        )
        scale = v55_scale.copy()
        scale[selected] = report["high_scale"]
        prediction = sigmoid(logit(v38) + scale * total_direction)
        report["pitcher_clustered_bootstrap"] = clustered_interval(
            target, v55, prediction, rows["pitcher_id"].to_numpy(), repeats=2000,
        )
    safe = [report for report in ranked[:20] if (
        report["gains"]["all"] > 0.
        and report["minimum_quarter_gain"] > 0.
        and report["minimum_roster_gain"] > 0.
        and report["selected_gain"] is not None
        and report["selected_gain"] > 0.
        and report["pitcher_clustered_bootstrap"]["p05"] > 0.
    )]
    safe = sorted(safe, key=lambda report: report["gains"]["all"], reverse=True)
    output = {
        "anchor": {"version": "v55", "public_score": 1113.6},
        "base_f_scale": BASE_F_SCALE,
        "safe": safe, "ranked": ranked,
        "row_independent": True, "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v56_v54_agreement.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe, "top": ranked[:12]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
