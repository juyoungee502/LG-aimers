"""Correct double counting in the decomposed failure probability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", usecols=["season", "game_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        v19 = {key: loaded[key] for key in loaded.files}
    reports = []
    for mode in ("product", "minimum_product", "geometric"):
        folds = {}
        for year in (2023, 2024):
            with np.load(
                root / f"research/failure_specialists_{year}_prior_context.npz"
            ) as loaded:
                stored = {key: loaded[key] for key in loaded.files}
            index = list(stored["variants"].astype(str)).index("uniform_depth8")
            matrix = stored["predictions"][index, :, :3].astype(np.float64)
            reverse, middle, wayoff = matrix.T
            raw = np.clip(1. - reverse - middle - wayoff, 1e-5, 1. - 1e-5)
            if mode == "product":
                intersection = reverse * middle
            elif mode == "minimum_product":
                # A conservative correlated-Bernoulli estimate between
                # independence and the Frechet upper bound.
                intersection = .5 * (reverse * middle + np.minimum(reverse, middle))
            else:
                # Damp the independence estimate in low-confidence regions.
                intersection = reverse * middle * np.sqrt(
                    np.clip((1. - wayoff) / .8, .5, 1.5)
                )
            corrected = np.clip(
                1. - reverse - middle - wayoff + intersection,
                1e-5, 1. - 1e-5,
            )
            fold = v19["season"] == year
            y = v19["target"][fold].astype(np.float64)
            base = v19["blended"][fold].astype(np.float64)
            if not np.allclose(y, stored["target"]):
                raise ValueError(f"Failure rows differ for {year}")
            regular = data.loc[data["season"].eq(year), "game_type"].eq("R").to_numpy()
            folds[year] = (y, base, raw, corrected, regular)

        for weight in np.arange(-.1, .401, .005):
            gains, halves = {}, []
            means = {}
            for year, (y, base, raw, corrected, regular) in folds.items():
                prediction = base.copy()
                prediction[regular] = sigmoid(
                    logit(base[regular])
                    + weight * (logit(corrected[regular]) - logit(raw[regular]))
                )
                midpoint = len(y) // 2
                values = [
                    bss(y, prediction) - bss(y, base),
                    bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                    bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                ]
                gains[str(year)] = values
                halves.extend(values[1:])
                means[str(year)] = {
                    "raw": float(raw[regular].mean()),
                    "corrected": float(corrected[regular].mean()),
                    "target": float(y[regular].mean()),
                }
            reports.append({
                "mode": mode, "weight": float(weight),
                "gain_2023": gains["2023"][0],
                "gain_2024": gains["2024"][0],
                "gain_2023_halves": gains["2023"][1:],
                "gain_2024_halves": gains["2024"][1:],
                "min_year": min(gains["2023"][0], gains["2024"][0]),
                "min_half": min(halves), "means": means,
            })
    reports.sort(
        key=lambda row: (row["min_year"], row["min_half"], row["gain_2024"]),
        reverse=True,
    )
    output = root / "research/failure_union_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:60], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
