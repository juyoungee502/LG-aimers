"""Robust blend screen for the v30 hierarchical-residual axis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
WEIGHTS = np.round(np.arange(0., 1.001, .025), 3)


def score_slices(target, prediction, regular):
    rows = np.arange(len(target))
    quarters = np.array_split(rows, 4)
    result = {
        "all": bss(target, prediction),
        "R": bss(target[regular], prediction[regular]),
        "F": bss(target[~regular], prediction[~regular]),
    }
    for index, active in enumerate(quarters, 1):
        result[f"q{index}"] = bss(target[active], prediction[active])
    return result


def blend_coefficients(target, base, candidate, regular):
    """Return exact BSS-gain quadratics for convex R/F blend weights."""
    error = target - base
    delta = candidate - base
    rows = np.arange(len(target))
    masks = {
        "all": np.ones(len(target), dtype=bool),
        "R": regular,
        "F": ~regular,
    }
    for index, active in enumerate(np.array_split(rows, 4), 1):
        mask = np.zeros(len(target), dtype=bool)
        mask[active] = True
        masks[f"q{index}"] = mask
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        rate = float(target[mask].mean())
        denominator = rate * (1. - rate)
        regular_mask = mask & regular
        futures_mask = mask & ~regular
        result[name] = {
            "b_R": float(np.sum(
                error[regular_mask] * delta[regular_mask]
            ) / count),
            "c_R": float(np.sum(delta[regular_mask] ** 2) / count),
            "b_F": float(np.sum(
                error[futures_mask] * delta[futures_mask]
            ) / count),
            "c_F": float(np.sum(delta[futures_mask] ** 2) / count),
            "denominator": denominator,
        }
    return result


def blend_scores(base_scores, coefficients, r_weight, f_weight):
    scores = {}
    gains = {}
    for segment, values in coefficients.items():
        mse_reduction = (
            2. * values["b_R"] * r_weight
            - values["c_R"] * r_weight ** 2
            + 2. * values["b_F"] * f_weight
            - values["c_F"] * f_weight ** 2
        )
        gain = 100000. * mse_reduction / values["denominator"]
        gains[segment] = gain
        scores[segment] = base_scores[segment] + gain
    return scores, gains


def main():
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as source:
        archive = {key: source[key] for key in source.files}

    years = {}
    for year in (2023, 2024):
        fold = archive["season"] == year
        with np.load(
            ROOT / f"research/v30_hierarchical_residual_{year}.npz"
        ) as source:
            v30 = {key: source[key] for key in source.files}
        target = archive["target"][fold].astype(float)
        if not np.allclose(target, v30["target"]):
            raise ValueError(f"v30 rows do not align with v23 for {year}")
        names = list(v30["prediction_names"].astype(str))
        predictions = {
            name: v30["predictions"][:, index].astype(float)
            for index, name in enumerate(names)
        }
        predictions["hierarchical_base"] = v30["base"].astype(float)
        years[year] = {
            "target": target,
            "v23": archive["blended"][fold].astype(float),
            "regular": v30["regular"].astype(bool),
            "predictions": predictions,
        }

    report = {"standalone": {}, "global": {}, "game_type": {}}
    for values in years.values():
        values["base_scores"] = score_slices(
            values["target"], values["v23"], values["regular"],
        )
    model_names = list(years[2023]["predictions"])
    for name in model_names:
        report["standalone"][name] = {
            str(year): score_slices(
                values["target"], values["predictions"][name],
                values["regular"],
            )
            for year, values in years.items()
        }
        coefficients = {
            year: blend_coefficients(
                values["target"], values["v23"],
                values["predictions"][name], values["regular"],
            )
            for year, values in years.items()
        }
        candidates = []
        for weight in WEIGHTS:
            scores = {}
            gains = {}
            for year, values in years.items():
                scores[str(year)], gains[str(year)] = blend_scores(
                    values["base_scores"], coefficients[year],
                    weight, weight,
                )
            candidates.append({
                "weight": float(weight), "scores": scores, "gains": gains,
                "worst_all_gain": min(gains[y]["all"] for y in gains),
                "mean_all_gain": float(np.mean([
                    gains[y]["all"] for y in gains
                ])),
                "worst_quarter_gain": min(
                    gains[y][f"q{quarter}"]
                    for y in gains for quarter in range(1, 5)
                ),
            })
        report["global"][name] = sorted(
            candidates,
            key=lambda row: (row["worst_all_gain"], row["mean_all_gain"]),
            reverse=True,
        )[:8]

        type_candidates = []
        for r_weight in WEIGHTS:
            for f_weight in WEIGHTS:
                gains = {}
                scores = {}
                for year, values in years.items():
                    scores[str(year)], gains[str(year)] = blend_scores(
                        values["base_scores"], coefficients[year],
                        r_weight, f_weight,
                    )
                type_candidates.append({
                    "r_weight": float(r_weight),
                    "f_weight": float(f_weight),
                    "scores": scores,
                    "gains": gains,
                    "worst_all_gain": min(gains[y]["all"] for y in gains),
                    "mean_all_gain": float(np.mean([
                        gains[y]["all"] for y in gains
                    ])),
                    "worst_type_gain": min(
                        gains[y][kind] for y in gains for kind in ("R", "F")
                    ),
                    "worst_quarter_gain": min(
                        gains[y][f"q{quarter}"]
                        for y in gains for quarter in range(1, 5)
                    ),
                })
        report["game_type"][name] = sorted(
            type_candidates,
            key=lambda row: (
                row["worst_all_gain"], row["worst_type_gain"],
                row["mean_all_gain"],
            ),
            reverse=True,
        )[:12]

    output = ROOT / "research/v30_blend_screen.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in model_names:
        global_best = report["global"][name][0]
        type_best = report["game_type"][name][0]
        print(
            f"{name}: global w={global_best['weight']:+.3f} "
            f"worst={global_best['worst_all_gain']:+.3f} "
            f"mean={global_best['mean_all_gain']:+.3f}; "
            f"type R={type_best['r_weight']:+.3f} "
            f"F={type_best['f_weight']:+.3f} "
            f"worst={type_best['worst_all_gain']:+.3f} "
            f"mean={type_best['mean_all_gain']:+.3f} "
            f"worst_type={type_best['worst_type_gain']:+.3f} "
            f"worst_q={type_best['worst_quarter_gain']:+.3f}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
