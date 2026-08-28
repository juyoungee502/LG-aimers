"""Audit the v56 R-only fine-pitch prior under 2024 roster turnover."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v53_roster_stability import candidate_score, clustered_interval


ROOT = Path(__file__).resolve().parent


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    rows = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    previous = raw.loc[raw["season"].eq(2023)]
    with np.load(ROOT / "research/v56_fine_pitch_failure_prior_2024.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["base"].astype(float)
        prediction = archive["prediction"].astype(float)
    if not (
        len(rows) == len(target)
        and np.allclose(target, rows["control_success"].to_numpy(float))
    ):
        raise ValueError("v56 and train rows do not align")

    game_type = rows["game_type"].astype(str).to_numpy()
    regular = game_type == "R"
    prior_pitchers = set(previous["pitcher_id"].astype(int))
    returning = rows["pitcher_id"].astype(int).isin(prior_pitchers).to_numpy()
    last_team = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["pitcher_team_id"]
    )
    prior_team = rows["pitcher_id"].map(last_team)
    same_team = (prior_team.eq(rows["pitcher_team_id"]) & returning).to_numpy()
    prior_end = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["asof_pitcher_n"] + 1.
    )
    origin = rows["pitcher_id"].map(prior_end).fillna(0.).to_numpy(float)
    exposure = np.maximum(0., rows["asof_pitcher_n"].to_numpy(float) - origin)
    cohorts = {
        "all": np.ones(len(rows), dtype=bool),
        "R": regular,
        "F": ~regular,
        "R_returning": regular & returning,
        "R_new": regular & ~returning,
        "R_same_team": regular & same_team,
        "R_team_change": regular & returning & ~same_team,
        "R_low_exposure": regular & (exposure <= 100.),
        "R_high_exposure": regular & (exposure > 100.),
    }
    base_scores = {
        name: candidate_score(target, base, active)
        for name, active in cohorts.items()
    }
    scores = {
        name: candidate_score(target, prediction, active)
        for name, active in cohorts.items()
    }
    gains = {
        name: scores[name] - base_scores[name]
        for name in cohorts if scores[name] is not None
    }
    pitcher_gains = {}
    for pitcher, index in rows.loc[regular].groupby(
        "pitcher_id", observed=True,
    ).groups.items():
        active = np.zeros(len(rows), dtype=bool)
        active[np.asarray(list(index), dtype=int)] = True
        if active.sum() >= 500:
            pitcher_gains[str(pitcher)] = float(
                bss(target[active], prediction[active])
                - bss(target[active], base[active])
            )
    bootstrap = clustered_interval(
        target[regular], base[regular], prediction[regular],
        rows.loc[regular, "pitcher_id"].to_numpy(), repeats=5000,
    )
    report = {
        "cohort_rows": {name: int(active.sum()) for name, active in cohorts.items()},
        "gains": gains,
        "minimum_returning_roster_gain": float(min(
            gains[name] for name in (
                "R_returning", "R_same_team", "R_team_change",
                "R_low_exposure", "R_high_exposure",
            ) if name in gains
        )),
        "pitcher_gains": pitcher_gains,
        "negative_pitcher_fraction": float(np.mean(
            np.asarray(list(pitcher_gains.values())) < 0.
        )),
        "clustered_bootstrap_R": bootstrap,
    }
    output = ROOT / "research/v56_roster_stability.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
