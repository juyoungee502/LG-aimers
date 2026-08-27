"""Jointly screen the v34/v35 low-cardinality components over recent bases."""
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


def blocks(length):
    position = np.arange(length)
    result = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index, part in enumerate(np.array_split(position, 4), 1):
        mask = np.zeros(length, dtype=bool)
        mask[part] = True
        result[f"q{index}"] = mask
    return result


def score_blocks(target, prediction, masks):
    return {name: float(bss(target[mask], prediction[mask]))
            for name, mask in masks.items()}


def main():
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = {key: archive[key] for key in archive.files}
    with np.load(
        ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
    ) as archive:
        direct = {key: archive[key] for key in archive.files}
    target = failure["target"].astype(float)
    failure_probability = failure["new_failure"].astype(float)
    direct_probability = direct["prediction"].astype(float)
    game_type = direct["game_type"].astype(str)
    if not np.allclose(target, direct["target"]):
        raise ValueError("v34/v35 validation rows differ")

    bases = {}
    # v26 is intentionally excluded despite its higher local score: its actual
    # public score (1079) was lower than v23 (1105), so ranking on it would
    # knowingly optimize a disproven local axis.  v24 is the conservative
    # forward candidate and v23 remains the public anchor.
    for version in (23, 24):
        with np.load(ROOT / f"outputs/v{version}_oof_predictions.npz") as archive:
            fold = archive["season"] == 2024
            if not np.allclose(target, archive["target"][fold]):
                raise ValueError(f"v{version} validation rows differ")
            bases[f"v{version}"] = np.clip(
                archive["blended"][fold].astype(float), .005, .995,
            )

    masks = blocks(len(target))
    reference_scores = {
        name: score_blocks(target, prediction, masks)
        for name, prediction in bases.items()
    }
    candidates = []
    for base_name, base in bases.items():
        base_logit = logit(base)
        for failure_weight in np.round(np.arange(0., .2501, .025), 4):
            first = sigmoid(
                (1. - failure_weight) * base_logit
                + failure_weight * logit(failure_probability)
            )
            for direct_gate in ("none", "all", "F"):
                direct_weights = (0.,) if direct_gate == "none" else np.round(
                    np.arange(.025, .2001, .025), 4,
                )
                for direct_weight in direct_weights:
                    candidate = first.copy()
                    if direct_gate != "none":
                        active = (
                            np.ones(len(target), dtype=bool)
                            if direct_gate == "all" else game_type == direct_gate
                        )
                        candidate[active] = sigmoid(
                            (1. - direct_weight) * logit(first[active])
                            + direct_weight * logit(direct_probability[active])
                        )
                    scores = score_blocks(target, candidate, masks)
                    gains_v23 = {
                        key: scores[key] - reference_scores["v23"][key]
                        for key in scores
                    }
                    gains_base = {
                        key: scores[key] - reference_scores[base_name][key]
                        for key in scores
                    }
                    regular = game_type == "R"
                    futures = ~regular
                    candidates.append({
                        "base": base_name,
                        "failure_weight": float(failure_weight),
                        "direct_gate": direct_gate,
                        "direct_weight": float(direct_weight),
                        "scores": scores,
                        "gains_v23": gains_v23,
                        "gains_base": gains_base,
                        "min_quarter_gain_v23": float(min(
                            gains_v23[f"q{i}"] for i in range(1, 5)
                        )),
                        "min_half_gain_v23": float(min(
                            gains_v23["h1"], gains_v23["h2"]
                        )),
                        "gain_R_v23": float(
                            bss(target[regular], candidate[regular])
                            - bss(target[regular], bases["v23"][regular])
                        ),
                        "gain_F_v23": float(
                            bss(target[futures], candidate[futures])
                            - bss(target[futures], bases["v23"][futures])
                        ),
                    })

    by_score = sorted(
        candidates,
        key=lambda row: (row["scores"]["all"], row["min_quarter_gain_v23"]),
        reverse=True,
    )
    by_strict = sorted(
        candidates,
        key=lambda row: (
            min(row["min_quarter_gain_v23"], row["min_half_gain_v23"]),
            row["scores"]["all"],
        ),
        reverse=True,
    )
    report = {
        "reference_scores": reference_scores,
        "base_policy": "v23 public anchor plus conservative v24 only; v26 rejected by public LB",
        "best_score": by_score[:100],
        "best_strict": by_strict[:100],
        "best_by_base": {
            base_name: [row for row in by_score if row["base"] == base_name][:50]
            for base_name in bases
        },
    }
    output = ROOT / "research/v36_joint_lowcard_blend.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "reference_scores": reference_scores,
        "best_score": by_score[:10],
        "best_strict": by_strict[:10],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
