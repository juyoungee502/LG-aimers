"""Re-audit historical model-stage increments on top of the final v23 base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


STAGES = tuple(range(16, 24))


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv",
        usecols=["season", "game_month", "game_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    loaded = {}
    for stage in STAGES:
        with np.load(root / f"outputs/v{stage}_oof_predictions.npz") as source:
            loaded[stage] = {key: source[key] for key in source.files}
    reference = loaded[23]
    for stage, source in loaded.items():
        if not np.array_equal(source["season"], reference["season"]):
            raise ValueError(f"v{stage} seasons do not align")
        if not np.allclose(source["target"], reference["target"]):
            raise ValueError(f"v{stage} targets do not align")

    reports = []
    for stage in STAGES[1:]:
        previous = loaded[stage - 1]["blended"].astype(float)
        current = loaded[stage]["blended"].astype(float)
        for geometry in ("probability", "logit"):
            direction = current - previous if geometry == "probability" else (
                logit(current) - logit(previous)
            )
            for scale in np.arange(-1.0, 1.001, .025):
                gains = {}
                temporal = []
                for year in (2023, 2024):
                    fold = reference["season"] == year
                    target = reference["target"][fold].astype(float)
                    base = reference["blended"][fold].astype(float)
                    rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
                    year_masks = masks(rows)
                    if geometry == "probability":
                        candidate = np.clip(base + scale * direction[fold], .005, .995)
                    else:
                        candidate = sigmoid(logit(base) + scale * direction[fold])
                    gains[str(year)] = {
                        name: bss(target[mask], candidate[mask])
                        - bss(target[mask], base[mask])
                        for name, mask in year_masks.items() if mask.any()
                    }
                    temporal.extend(
                        value for name, value in gains[str(year)].items()
                        if name != "all"
                    )
                reports.append({
                    "stage": stage, "geometry": geometry, "scale": float(scale),
                    "gains": gains,
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "min_segment": min(temporal),
                    "mean_year": np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ]),
                })
    nonzero = [row for row in reports if abs(row["scale"]) > 1e-8]
    rankings = {
        "maximin_segment": sorted(
            nonzero,
            key=lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
            reverse=True,
        )[:100],
        "maximin_year": sorted(
            nonzero,
            key=lambda row: (row["min_year"], row["min_segment"], row["mean_year"]),
            reverse=True,
        )[:100],
        "best_mean": sorted(
            nonzero,
            key=lambda row: (row["mean_year"], row["min_year"], row["min_segment"]),
            reverse=True,
        )[:100],
    }
    output = root / "research/v23_stage_reweight.json"
    output.write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    print(json.dumps({name: values[:15] for name, values in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
