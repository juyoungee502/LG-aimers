"""Diagnose mean-free resolution in older auxiliary-task specialists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv", usecols=["season", "game_month", "game_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as z:
        v23 = {key: z[key] for key in z.files}
    reports = []
    for year in (2023, 2024):
        fold = v23["season"] == year
        target = v23["target"][fold].astype(float)
        base = v23["blended"][fold].astype(float)
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        year_masks = masks(rows)
        specialists = {}
        with np.load(root / f"research/outcome_moe_{year}.npz") as z:
            specialists["outcome_moe"] = z["mixture"].astype(float)
        with np.load(root / f"research/multitask_catboost_{year}.npz") as z:
            for index, name in enumerate(z["variants"].astype(str)):
                specialists[f"multitask_{name}"] = z["predictions"][:, index].astype(float)
        for specialist_name, prediction in specialists.items():
            regular = year_masks["regular"]
            for geometry, raw_direction in (
                ("probability", prediction - prediction[regular].mean()),
                ("logit", logit(prediction) - logit(prediction)[regular].mean()),
            ):
                direction = np.zeros(len(target), dtype=float)
                direction[regular] = raw_direction[regular]
                for weight in np.arange(-.50, .501, .025):
                    if geometry == "probability":
                        candidate = np.clip(base + weight * direction, .005, .995)
                    else:
                        candidate = sigmoid(logit(base) + weight * direction)
                    gains = {
                        name: bss(target[mask], candidate[mask]) - bss(
                            target[mask], base[mask],
                        )
                        for name, mask in year_masks.items()
                    }
                    reports.append({
                        "year": year, "specialist": specialist_name,
                        "geometry": geometry, "weight": float(weight), "gains": gains,
                        "min_temporal": min(
                            gains["first_half"], gains["second_half"],
                            gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                        ),
                    })
    paired = []
    keys = {
        (row["specialist"], row["geometry"], round(row["weight"], 4))
        for row in reports
    }
    for specialist, geometry, weight in keys:
        selected = [
            row for row in reports
            if row["specialist"] == specialist and row["geometry"] == geometry
            and round(row["weight"], 4) == weight
        ]
        if len(selected) != 2:
            continue
        by_year = {str(row["year"]): row["gains"] for row in selected}
        paired.append({
            "specialist": specialist, "geometry": geometry, "weight": weight,
            "gains": by_year,
            "min_temporal": min(row["min_temporal"] for row in selected),
            "min_year": min(row["gains"]["all"] for row in selected),
            "mean_year": np.mean([row["gains"]["all"] for row in selected]),
        })
    paired.sort(
        key=lambda row: (row["min_temporal"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research/v23_legacy_resolution.json"
    output.write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(json.dumps({"top": paired[:80]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
