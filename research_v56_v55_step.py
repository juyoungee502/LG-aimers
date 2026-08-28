"""Validate the next conservative F-scaling step from public-positive v55."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v53_roster_stability import clustered_interval


ROOT = Path(__file__).resolve().parent
V55_F_SCALE = 1.125
V56_F_SCALE = 1.25


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
    v55 = sigmoid(logit(v38) + np.where(futures, V55_F_SCALE, 1.) * direction)
    v56 = sigmoid(logit(v38) + np.where(futures, V56_F_SCALE, 1.) * direction)

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
    gains = {
        name: float(bss(target[active], v56[active]) - bss(target[active], v55[active]))
        for name, active in cohorts.items() if int(active.sum()) >= 500
    }
    bootstrap = clustered_interval(
        target, v55, v56, rows["pitcher_id"].to_numpy(), repeats=5000,
    )
    if (
        gains["all"] <= 0.
        or min(gains[f"q{i}"] for i in range(1, 5)) <= 0.
        or min(gains[name] for name in (
            "returning_both", "roster_change", "same_teams",
            "player_or_team_change",
        )) <= 0.
    ):
        raise RuntimeError(f"v56 promotion gate failed: {gains}")
    output = {
        "anchor": {"version": "v55", "public_score": 1113.6},
        "v55_f_scale": V55_F_SCALE, "v56_f_scale": V56_F_SCALE,
        "gains": gains,
        "minimum_quarter_gain": min(gains[f"q{i}"] for i in range(1, 5)),
        "minimum_roster_gain": min(gains[name] for name in (
            "returning_both", "roster_change", "same_teams",
            "player_or_team_change",
        )),
        "pitcher_clustered_bootstrap": bootstrap,
        "row_independent": True, "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v56_v55_step.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
