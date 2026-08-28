"""Gate extra public-positive v54 directions by current-season exposure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v58_fraction_roster_stability import (
    clustered_interval, pitcher_season_exposure, roster_masks, score,
)


ROOT = Path(__file__).resolve().parent
NAMES = ("command", "overlap", "recent", "joint")
POLICIES = {
    "command_half": (.05, 0., 0., 0.),
    "command_full": (.10, 0., 0., 0.),
    "overlap_half": (0., .0375, 0., 0.),
    "overlap_full": (0., .075, 0., 0.),
    "recent_half": (0., 0., .025, 0.),
    "recent_full": (0., 0., .05, 0.),
    "joint_half": (0., 0., 0., .009375),
    "joint_full": (0., 0., 0., .01875),
    "command_recent_half": (.05, 0., .025, 0.),
    "overlap_recent_half": (0., .0375, .025, 0.),
    "command_overlap_half": (.05, .0375, 0., 0.),
    "recent_joint_half": (0., 0., .025, .009375),
    "f_components_half": (.05, .0375, .025, 0.),
    "all_components_half": (.05, .0375, .025, .009375),
}
THRESHOLDS = (25., 50., 100., 150., 200., 300., 500.)


def metrics(target, base, prediction, segments):
    gains = {}
    for name, active in segments.items():
        before = score(target, base, active)
        after = score(target, prediction, active)
        if before is not None and after is not None:
            gains[name] = after - before
    affected_roster = (
        "F_returning_both", "F_roster_change", "F_same_teams",
        "F_player_or_team_change", "F_high_pitcher_exposure",
    )
    return {
        "gain_all": float(gains["all"]),
        "gain_R": float(gains["R"]),
        "gain_F": float(gains["F"]),
        "quarter_gains": [float(gains[f"q{i}"]) for i in range(1, 5)],
        "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
        "affected_roster_gains": {
            name: float(gains[name]) for name in affected_roster if name in gains
        },
        "minimum_affected_roster_gain": float(min(
            gains[name] for name in affected_roster if name in gains
        )),
        "low_exposure_gain": float(gains["F_low_pitcher_exposure"]),
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

    corrected = np.clip(failure + .45 * overlap, .005, .995)
    recent = np.clip(.5 * (recent_a + recent_b), .005, .995)
    directions = np.zeros((4, len(target)), dtype=float)
    directions[0, futures] = logit(command[futures]) - logit(v38[futures])
    directions[1, futures] = logit(corrected[futures]) - logit(v38[futures])
    directions[2, futures] = logit(recent[futures]) - logit(v38[futures])
    directions[3] = logit(joint) - logit(v38)
    exposure = pitcher_season_exposure(raw, rows, year)
    segments = {
        **masks(len(rows)), "R": ~futures, "F": futures,
        **roster_masks(raw, rows, year),
    }

    reports = []
    for threshold in THRESHOLDS:
        selected = futures & (exposure > threshold)
        for policy, values in POLICIES.items():
            weights = np.asarray(values, dtype=float)
            correction = weights @ directions
            prediction = v54.copy()
            prediction[selected] = sigmoid(
                logit(v54[selected]) + correction[selected]
            )
            report = metrics(target, v54, prediction, segments)
            report.update({
                "policy": policy,
                "threshold": threshold,
                "selected_rows": int(selected.sum()),
                "extra_weights": dict(zip(NAMES, map(float, weights))),
            })
            reports.append(report)

    by_score = sorted(reports, key=lambda row: row["gain_all"], reverse=True)
    by_robustness = sorted(
        reports,
        key=lambda row: (
            row["minimum_quarter_gain"],
            row["minimum_affected_roster_gain"], row["gain_all"],
        ),
        reverse=True,
    )
    audit_rows = {id(row): row for row in [*by_score[:25], *by_robustness[:25]]}
    for row in audit_rows.values():
        selected = futures & (exposure > row["threshold"])
        weights = np.fromiter(row["extra_weights"].values(), dtype=float)
        correction = weights @ directions
        prediction = v54.copy()
        prediction[selected] = sigmoid(
            logit(v54[selected]) + correction[selected]
        )
        row["clustered_bootstrap"] = clustered_interval(
            target, v54, prediction, rows["pitcher_id"].to_numpy(), repeats=2000,
        )
    safe = sorted([
        row for row in audit_rows.values()
        if row["gain_all"] > 0.
        and row["minimum_quarter_gain"] > 0.
        and row["minimum_affected_roster_gain"] > 0.
        and row["clustered_bootstrap"]["p05"] > 0.
    ], key=lambda row: row["gain_all"], reverse=True)
    report = {
        "year": year,
        "v54_bss": float(bss(target, v54)),
        "safe_by_score": safe,
        "best_score": by_score[:25],
        "best_robustness": by_robustness[:25],
        "public_anchor": {"version": "v54", "score": 1113},
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v64_v54_exposure_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "v54_bss": report["v54_bss"], "safe_by_score": safe[:12],
        "best_robustness": by_robustness[:8],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
