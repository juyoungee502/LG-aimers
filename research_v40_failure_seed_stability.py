"""Compare one- and three-seed failure ensembles inside the v38 blend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def masks(length):
    position = np.arange(length)
    output = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index, part in enumerate(np.array_split(position, 4), 1):
        active = np.zeros(length, dtype=bool)
        active[part] = True
        output[f"q{index}"] = active
    return output


def main():
    failures = {}
    for label, filename in {
        "s1": "v34_categorical_failure_lowcard_no_ids_hl2_2024.npz",
        "s3": "v34_categorical_failure_lowcard_no_ids_hl2_s3_2024.npz",
    }.items():
        with np.load(ROOT / "research" / filename) as archive:
            failures[label] = {key: archive[key] for key in archive.files}
    with np.load(
        ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
    ) as archive:
        direct = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        public_anchor = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not all(np.allclose(target, item["target"]) for item in failures.values()):
        raise ValueError("Failure and v24 rows differ")
    if not np.allclose(target, direct["target"]):
        raise ValueError("Direct and v24 rows differ")

    blocks = masks(len(target))
    v23_scores = {
        name: float(bss(target[active], public_anchor[active]))
        for name, active in blocks.items()
    }
    game_type = direct["game_type"].astype(str)
    reports = []
    for seeds, item in failures.items():
        failure = item["new_failure"].astype(float)
        for failure_weight in np.round(np.arange(.075, .2251, .0125), 4):
            first = sigmoid(
                (1. - failure_weight) * logit(base)
                + failure_weight * logit(failure)
            )
            for direct_weight in np.round(np.arange(.05, .1501, .0125), 4):
                prediction = sigmoid(
                    (1. - direct_weight) * logit(first)
                    + direct_weight * logit(direct["prediction"])
                )
                scores = {
                    name: float(bss(target[active], prediction[active]))
                    for name, active in blocks.items()
                }
                gains = {name: scores[name] - v23_scores[name] for name in scores}
                regular = game_type == "R"
                reports.append({
                    "seeds": seeds,
                    "failure_weight": float(failure_weight),
                    "direct_weight": float(direct_weight),
                    "scores": scores,
                    "gains_v23": gains,
                    "min_quarter_gain_v23": float(min(
                        gains[f"q{i}"] for i in range(1, 5)
                    )),
                    "min_half_gain_v23": float(min(gains["h1"], gains["h2"])),
                    "gain_R_v23": float(
                        bss(target[regular], prediction[regular])
                        - bss(target[regular], public_anchor[regular])
                    ),
                    "gain_F_v23": float(
                        bss(target[~regular], prediction[~regular])
                        - bss(target[~regular], public_anchor[~regular])
                    ),
                })
    by_score = sorted(
        reports,
        key=lambda row: (row["scores"]["all"], row["min_quarter_gain_v23"]),
        reverse=True,
    )
    by_robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter_gain_v23"], row["min_half_gain_v23"]),
            row["scores"]["all"],
        ),
        reverse=True,
    )
    report = {
        "v38_frozen": {
            "failure_weight": .175, "direct_weight": .10,
        },
        "best_score_by_seeds": {
            seeds: next(row for row in by_score if row["seeds"] == seeds)
            for seeds in failures
        },
        "best_robust_by_seeds": {
            seeds: next(row for row in by_robust if row["seeds"] == seeds)
            for seeds in failures
        },
        "frozen_by_seeds": {
            seeds: next(
                row for row in reports
                if row["seeds"] == seeds
                and row["failure_weight"] == .175
                and row["direct_weight"] == .10
            )
            for seeds in failures
        },
    }
    output = ROOT / "research/v40_failure_seed_stability.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
