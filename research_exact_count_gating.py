"""Find exact-count expert weights that transfer across both rolling years."""
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


def gains(y, base, prediction):
    midpoint = len(y) // 2
    return [
        bss(y, prediction) - bss(y, base),
        bss(y[:midpoint], prediction[:midpoint])
        - bss(y[:midpoint], base[:midpoint]),
        bss(y[midpoint:], prediction[midpoint:])
        - bss(y[midpoint:], base[midpoint:]),
    ]


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv",
        usecols=["season", "game_type", "balls_before", "strikes_before"],
        encoding="utf-8-sig", low_memory=False,
    )
    folds = {}
    for year in (2023, 2024):
        with np.load(root / f"research/exact_count_specialist_{year}.npz") as loaded:
            data = {key: loaded[key] for key in loaded.files}
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        folds[year] = {
            "y": data["target"].astype(np.float64),
            "base": data["base"].astype(np.float64),
            "candidate": data["prediction"].astype(np.float64),
            "regular": rows["game_type"].eq("R").to_numpy(),
            "count": (
                rows["balls_before"].to_numpy(np.int8) * 3
                + rows["strikes_before"].to_numpy(np.int8)
            ),
        }

    selected = np.zeros(12, dtype=np.float64)
    count_reports = []
    for count in range(12):
        trials = []
        for weight in np.arange(-.30, .301, .01):
            yearly, halves = {}, []
            for year, fold in folds.items():
                mask = fold["regular"] & (fold["count"] == count)
                prediction = fold["base"].copy()
                prediction[mask] = sigmoid(
                    (1. - weight) * logit(fold["base"][mask])
                    + weight * logit(fold["candidate"][mask])
                )
                values = gains(fold["y"], fold["base"], prediction)
                yearly[str(year)] = values
                halves.extend(values[1:])
            trials.append({
                "count": count, "weight": float(weight),
                "gain_2023": yearly["2023"][0],
                "gain_2024": yearly["2024"][0],
                "halves_2023": yearly["2023"][1:],
                "halves_2024": yearly["2024"][1:],
                "min_year": min(yearly["2023"][0], yearly["2024"][0]),
                "min_half": min(halves),
                "mean_half": float(np.mean(halves)),
            })
        # The worst half is the primary guard.  Zero is always available, so
        # a selected nonzero weight must improve every temporal slice.
        trials.sort(
            key=lambda row: (row["min_half"], row["min_year"], row["mean_half"]),
            reverse=True,
        )
        best = trials[0]
        selected[count] = best["weight"] if best["min_half"] > 0. else 0.
        count_reports.append(best)

    combined = {}
    for shrinkage in (.5, .75, 1.0):
        report = {}
        for year, fold in folds.items():
            prediction = fold["base"].copy()
            for count, raw_weight in enumerate(selected):
                weight = shrinkage * raw_weight
                if weight == 0.:
                    continue
                mask = fold["regular"] & (fold["count"] == count)
                prediction[mask] = sigmoid(
                    (1. - weight) * logit(fold["base"][mask])
                    + weight * logit(fold["candidate"][mask])
                )
            report[str(year)] = gains(fold["y"], fold["base"], prediction)
        combined[str(shrinkage)] = report

    output = {
        "selected_weights": selected.tolist(),
        "per_count": count_reports,
        "combined": combined,
    }
    path = root / "research/exact_count_gating_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
