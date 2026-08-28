"""Re-audit only the v54 directions that have a positive public anchor."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v58_fraction_roster_stability import (
    clustered_interval, roster_masks, score,
)


ROOT = Path(__file__).resolve().parent
NAMES = ("command", "overlap", "recent", "joint")
BASE_WEIGHTS = np.asarray([.10, .075, .05, .01875], dtype=float)
MULTIPLIERS = (0., .5, 1., 1.5, 2.)


def candidate_metrics(target, base, prediction, segments):
    gains = {}
    for name, active in segments.items():
        base_score = score(target, base, active)
        prediction_score = score(target, prediction, active)
        if base_score is not None and prediction_score is not None:
            gains[name] = prediction_score - base_score
    return {
        "gain_all": float(gains["all"]),
        "gain_R": float(gains["R"]),
        "gain_F": float(gains["F"]),
        "quarter_gains": [float(gains[f"q{i}"]) for i in range(1, 5)],
        "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
        "roster_gains": {
            name: float(gains[name]) for name in (
                "F_returning_both", "F_roster_change", "F_same_teams",
                "F_player_or_team_change", "F_low_pitcher_exposure",
                "F_high_pitcher_exposure",
            ) if name in gains
        },
        "minimum_roster_gain": float(min(
            gains[name] for name in (
                "F_returning_both", "F_roster_change", "F_same_teams",
                "F_player_or_team_change", "F_low_pitcher_exposure",
                "F_high_pitcher_exposure",
            ) if name in gains
        )),
    }


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    year = 2024
    rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    target = rows["control_success"].to_numpy(float)
    futures = rows["game_type"].astype(str).eq("F").to_numpy()
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == year
        v38 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        active = archive["season"] == year
        v54 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
    with np.load(ROOT / "research/v43_multiclass_hl2_s1_2024.npz") as archive:
        command = archive["prediction"].astype(float)
    with np.load(ROOT / "research/v45_overlap_hl2_2024.npz") as archive:
        overlap = archive["overlap_probability"].astype(float)
    with np.load(ROOT / "research/v48_regime_command_s3_2024.npz") as archive:
        recent_a = archive["multiclass"].astype(float)
    with np.load(
        ROOT / "research/v49_regime_multiclass_complexity_2024.npz"
    ) as archive:
        recent_b = archive["d6_i1000"].astype(float)
    with np.load(
        ROOT / "research/v52_pitch_command_joint_s3_2024.npz"
    ) as archive:
        joint = archive["history_no_team"].astype(float)
    arrays = (v38, v54, failure, command, overlap, recent_a, recent_b, joint)
    if not all(len(value) == len(target) for value in arrays):
        raise ValueError("Frozen v54 arrays do not align")

    corrected = np.clip(failure + .45 * overlap, .005, .995)
    recent = np.clip(.5 * (recent_a + recent_b), .005, .995)
    directions = np.zeros((4, len(target)), dtype=float)
    directions[0, futures] = logit(command[futures]) - logit(v38[futures])
    directions[1, futures] = logit(corrected[futures]) - logit(v38[futures])
    directions[2, futures] = logit(recent[futures]) - logit(v38[futures])
    directions[3] = logit(joint) - logit(v38)
    reconstructed = sigmoid(logit(v38) + BASE_WEIGHTS @ directions)
    maximum_drift = float(np.max(np.abs(reconstructed - v54)))
    if maximum_drift > 2e-6:
        raise ValueError(f"v54 reconstruction drift: {maximum_drift}")

    segments = {
        **masks(len(rows)), "R": ~futures, "F": futures,
        **roster_masks(raw, rows, year),
    }
    reports = []
    for multipliers in itertools.product(MULTIPLIERS, repeat=4):
        weights = BASE_WEIGHTS * np.asarray(multipliers, dtype=float)
        prediction = sigmoid(logit(v38) + weights @ directions)
        report = candidate_metrics(target, v54, prediction, segments)
        report["multipliers"] = dict(zip(NAMES, map(float, multipliers)))
        report["weights"] = dict(zip(NAMES, map(float, weights)))
        reports.append(report)

    by_score = sorted(reports, key=lambda row: row["gain_all"], reverse=True)
    by_robustness = sorted(
        reports,
        key=lambda row: (
            row["minimum_quarter_gain"], row["minimum_roster_gain"],
            row["gain_all"],
        ),
        reverse=True,
    )
    audit_rows = {id(row): row for row in [*by_score[:30], *by_robustness[:30]]}
    for row in audit_rows.values():
        weights = np.fromiter(row["weights"].values(), dtype=float)
        prediction = sigmoid(logit(v38) + weights @ directions)
        row["clustered_bootstrap"] = clustered_interval(
            target, v54, prediction, rows["pitcher_id"].to_numpy(), repeats=2000,
        )
    safe = sorted([
        row for row in audit_rows.values()
        if row["gain_all"] > 0.
        and row["minimum_quarter_gain"] > 0.
        and row["minimum_roster_gain"] > 0.
        and row["clustered_bootstrap"]["p05"] > 0.
    ], key=lambda row: row["gain_all"], reverse=True)
    report = {
        "year": year,
        "v38_bss": float(bss(target, v38)),
        "v54_bss": float(bss(target, v54)),
        "reconstruction_maximum_drift": maximum_drift,
        "safe_by_score": safe,
        "best_score": by_score[:30],
        "best_robustness": by_robustness[:30],
        "public_anchors": {"v54": 1113, "v60": 1110},
        "selection_policy": (
            "Search only frozen v54 component directions; require positive "
            "2024 quarters, roster cohorts, and pitcher-cluster bootstrap p05."
        ),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v63_v54_weight_frontier.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "v38_bss": report["v38_bss"], "v54_bss": report["v54_bss"],
        "safe_by_score": safe[:12],
        "best_robustness": by_robustness[:8],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
