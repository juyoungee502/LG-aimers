"""Jointly audit the v43 multiclass and v45 overlap-aware F directions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent


def score(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    regular = game_type == "R"
    result["R"] = float(bss(target[regular], prediction[regular]))
    result["F"] = float(bss(target[~regular], prediction[~regular]))
    return result


def main():
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        target = archive["target"][active].astype(float)
        v24 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
    with np.load(
        ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
    ) as archive:
        direct = archive["prediction"].astype(float)
        game_type = archive["game_type"].astype(str)
    with np.load(
        ROOT / "research/v43_multiclass_hl2_s1_2024.npz", allow_pickle=True,
    ) as archive:
        multiclass = archive["prediction"].astype(float)
    with np.load(
        ROOT / "research/v45_overlap_hl2_2024.npz", allow_pickle=True,
    ) as archive:
        overlap = archive["overlap_probability"].astype(float)

    first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
    v38 = sigmoid(.90 * logit(first) + .10 * logit(direct))
    active_f = game_type == "F"
    blocks = masks(len(target))
    baseline = score(target, v38, blocks, game_type)
    multi_direction = logit(multiclass) - logit(v38)

    reports = []
    for inclusion_scale in np.round(np.arange(.30, .601, .05), 3):
        corrected = np.clip(
            failure + inclusion_scale * overlap, .005, .995,
        )
        overlap_direction = logit(corrected) - logit(v38)
        for multi_weight in np.round(np.arange(0., .151, .0125), 4):
            for overlap_weight in np.round(np.arange(0., .101, .0125), 4):
                prediction = v38.copy()
                prediction[active_f] = sigmoid(
                    logit(v38[active_f])
                    + multi_weight * multi_direction[active_f]
                    + overlap_weight * overlap_direction[active_f]
                )
                result = score(target, prediction, blocks, game_type)
                gains = {name: result[name] - baseline[name] for name in result}
                reports.append({
                    "inclusion_scale": float(inclusion_scale),
                    "multi_weight": float(multi_weight),
                    "overlap_weight": float(overlap_weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })
    frozen = next(
        row for row in reports
        if row["inclusion_scale"] == .45
        and row["multi_weight"] == .10
        and row["overlap_weight"] == .075
    )
    by_score = sorted(reports, key=lambda row: row["scores"]["all"], reverse=True)
    by_robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"], row["gains"]["F"]),
            row["scores"]["all"],
        ), reverse=True,
    )
    report = {
        "v38": baseline,
        "direction_correlation_F": float(np.corrcoef(
            multi_direction[active_f], overlap_direction[active_f]
        )[0, 1]),
        "frozen_independent_best_combination": frozen,
        "best_score": by_score[:50],
        "best_robust": by_robust[:50],
    }
    output = ROOT / "research/v46_command_portfolio.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "v38": baseline,
        "direction_correlation_F": report["direction_correlation_F"],
        "frozen": frozen,
        "best_score": by_score[:10],
        "best_robust": by_robust[:10],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
