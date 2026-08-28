"""Audit variance-reduced v54 components using independent CatBoost seeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v53_roster_stability import clustered_interval


ROOT = Path(__file__).resolve().parent


def load(name, key):
    with np.load(ROOT / "research" / f"{name}.npz") as archive:
        return archive[key].astype(float)


def score(target, prediction, active):
    return float(bss(target[active], prediction[active]))


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    rows = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    target = rows["control_success"].to_numpy(float)
    game_type = rows["game_type"].astype(str).to_numpy()
    futures = game_type == "F"
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = np.clip(
            archive["blended"][archive["season"] == 2024].astype(float),
            .005, .995,
        )
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = np.clip(
            archive["blended"][archive["season"] == 2024].astype(float),
            .005, .995,
        )

    failure_1 = load("v34_categorical_failure_lowcard_no_ids_hl2_2024", "new_failure")
    failure_3 = load("v34_categorical_failure_lowcard_no_ids_hl2_s3_2024", "new_failure")
    command_1 = load("v43_multiclass_hl2_s1_2024", "prediction")
    command_3 = load("v43_multiclass_hl2_s3_2024", "prediction")
    overlap = load("v45_overlap_hl2_2024", "overlap_probability")
    recent_1 = load("v48_regime_command_2024", "multiclass")
    recent_3 = load("v48_regime_command_s3_2024", "multiclass")
    recent_complex = load("v49_regime_multiclass_complexity_2024", "d6_i1000")
    joint_1 = load("v52_pitch_command_joint_s1_2024", "history_no_team")
    joint_3 = load("v52_pitch_command_joint_s3_2024", "history_no_team")

    def assemble(failure, command, recent, joint):
        corrected = np.clip(failure + .45 * overlap, .005, .995)
        prediction = v38.copy()
        prediction[futures] = sigmoid(
            logit(v38[futures])
            + .10 * (logit(command[futures]) - logit(v38[futures]))
            + .075 * (logit(corrected[futures]) - logit(v38[futures]))
            + .05 * (logit(recent[futures]) - logit(v38[futures]))
        )
        return sigmoid(
            logit(prediction) + .01875 * (logit(joint) - logit(v38))
        )

    components = {
        "failure": (failure_1, .5 * (failure_1 + failure_3)),
        "command": (command_1, .5 * (command_1 + command_3)),
        "recent": (
            .5 * (recent_3 + recent_complex),
            (recent_1 + recent_3 + recent_complex) / 3.,
        ),
        "joint": (joint_3, .5 * (joint_1 + joint_3)),
    }
    original = assemble(*(components[name][0] for name in components))
    if np.max(np.abs(original - v54)) > 2e-5:
        raise ValueError(
            f"Could not reconstruct v54: max_abs={np.max(np.abs(original - v54))}"
        )
    candidates = {"v54": v54}
    for bits in range(1, 16):
        names = [name for index, name in enumerate(components) if bits & (1 << index)]
        values = [
            components[name][1 if name in names else 0]
            for name in components
        ]
        candidates["average_" + "_".join(names)] = assemble(*values)

    previous = raw.loc[raw["season"].eq(2023)]
    returning_pitcher = rows["pitcher_id"].isin(previous["pitcher_id"]).to_numpy()
    returning_batter = rows["batter_id"].isin(previous["batter_id"]).to_numpy()
    cohorts = {**masks(len(rows)), "R": ~futures, "F": futures}
    cohorts.update({
        "returning_both": returning_pitcher & returning_batter,
        "roster_change": ~(returning_pitcher & returning_batter),
    })
    base_scores = {
        name: score(target, v54, active) for name, active in cohorts.items()
        if int(active.sum()) >= 500
    }
    reports = []
    for name, prediction in candidates.items():
        gains = {
            cohort: score(target, prediction, active) - base_scores[cohort]
            for cohort, active in cohorts.items() if cohort in base_scores
        }
        bootstrap = clustered_interval(
            target, v54, prediction, rows["pitcher_id"].to_numpy(), repeats=2000,
        )
        reports.append({
            "name": name,
            "gains": gains,
            "minimum_quarter_gain": min(gains[f"q{i}"] for i in range(1, 5)),
            "minimum_roster_gain": min(gains["returning_both"], gains["roster_change"]),
            "pitcher_clustered_bootstrap": bootstrap,
        })
    ranked = sorted(reports, key=lambda row: (
        row["minimum_quarter_gain"], row["minimum_roster_gain"],
        row["pitcher_clustered_bootstrap"]["p05"], row["gains"]["all"],
    ), reverse=True)
    safe = [row for row in ranked if (
        row["gains"]["all"] > 0.
        and row["minimum_quarter_gain"] > 0.
        and row["minimum_roster_gain"] > 0.
        and row["pitcher_clustered_bootstrap"]["p05"] > 0.
    )]
    output = {
        "anchor": "v54", "valid_year": 2024,
        "safe": safe, "ranked": ranked,
        "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v55_seed_ensemble.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe, "top": ranked[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
