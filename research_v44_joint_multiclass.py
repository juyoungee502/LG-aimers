"""Screen the v43 coherent multiclass model inside the v38 ensemble."""
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


def describe(model, mode, gate, weight, target, candidate, baseline, blocks, game_type):
    scores = score(target, candidate, blocks, game_type)
    gains = {name: scores[name] - baseline[name] for name in scores}
    return {
        "model": model, "mode": mode, "gate": gate, "weight": float(weight),
        "scores": scores, "gains_v38": gains,
        "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
        "min_half": float(min(gains["h1"], gains["h2"])),
    }


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
    multiclasses = {}
    for name in ("s1", "s3"):
        with np.load(
            ROOT / f"research/v43_multiclass_hl2_{name}_2024.npz",
            allow_pickle=True,
        ) as archive:
            multiclasses[name] = archive["prediction"].astype(float)
            if not np.allclose(target, archive["target"]):
                raise ValueError(f"v43 {name} rows do not align")

    first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
    v38 = sigmoid(.90 * logit(first) + .10 * logit(direct))
    blocks = masks(len(target))
    baseline = score(target, v38, blocks, game_type)
    reports = []

    for model_name, multiclass in multiclasses.items():
        # Add an independent multiclass direction after the full v38 blend.
        direction = logit(multiclass) - logit(v38)
        for gate in ("all", "R", "F"):
            selected = np.ones(len(target), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(-.10, .301, .0125), 4):
                candidate = v38.copy()
                candidate[selected] = sigmoid(
                    logit(v38[selected]) + weight * direction[selected]
                )
                reports.append(describe(
                    model_name, "add_final", gate, weight, target, candidate,
                    baseline, blocks, game_type,
                ))

        # Replace a fraction of the independent-failure logit before the direct model.
        for gate in ("all", "R", "F"):
            selected = np.ones(len(target), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(0., 1.001, .05), 4):
                mixed_failure = failure.copy()
                mixed_failure[selected] = sigmoid(
                    (1. - weight) * logit(failure[selected])
                    + weight * logit(multiclass[selected])
                )
                candidate_first = sigmoid(
                    .825 * logit(v24) + .175 * logit(mixed_failure)
                )
                candidate = sigmoid(
                    .90 * logit(candidate_first) + .10 * logit(direct)
                )
                reports.append(describe(
                    model_name, "replace_failure", gate, weight, target, candidate,
                    baseline, blocks, game_type,
                ))

    by_score = sorted(
        reports, key=lambda row: (row["scores"]["all"], row["min_quarter"]),
        reverse=True,
    )
    by_robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"],
                row["gains_v38"]["R"], row["gains_v38"]["F"]),
            row["scores"]["all"],
        ), reverse=True,
    )
    report = {
        "v38": baseline,
        "correlations": {
            name: {
                "v38": float(np.corrcoef(value, v38)[0, 1]),
                "failure": float(np.corrcoef(value, failure)[0, 1]),
                "direct": float(np.corrcoef(value, direct)[0, 1]),
            } for name, value in multiclasses.items()
        },
        "best_score": by_score[:50],
        "best_robust": by_robust[:50],
    }
    output = ROOT / "research/v44_joint_multiclass.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "v38": baseline,
        "correlations": report["correlations"],
        "best_score": by_score[:10],
        "best_robust": by_robust[:10],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
