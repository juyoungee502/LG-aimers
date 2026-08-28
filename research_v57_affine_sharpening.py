"""Audit a row-independent affine spread correction for the v56 candidate.

The correction is deliberately restricted to regular-season (R) rows, which
v54-v56 leave unchanged.  For each forward fold, the centre is the target mean
of strictly earlier seasons.  Production therefore uses the 2019-2024 training
mean, never any evaluation-row statistic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import masks


ROOT = Path(__file__).resolve().parent
F_SCALE = 1.25


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    reference = float(np.mean(target) * (1.0 - np.mean(target)))
    return 100_000.0 * (1.0 - float(np.mean((target - prediction) ** 2)) / reference)


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def score_slices(target: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                 season: np.ndarray, game_type: np.ndarray) -> dict[str, float]:
    report: dict[str, float] = {}
    for year in (2023, 2024):
        year_rows = season == year
        order = np.flatnonzero(year_rows)
        parts = {
            "all": year_rows,
            "R": year_rows & (game_type == "R"),
            "F": year_rows & (game_type == "F"),
        }
        for quarter, split in enumerate(np.array_split(order, 4), start=1):
            selected = np.zeros(len(target), dtype=bool)
            selected[split] = True
            parts[f"q{quarter}"] = selected
        for name, rows in parts.items():
            if rows.sum() > 1:
                report[f"{year}/{name}"] = bss(target[rows], candidate[rows]) - bss(
                    target[rows], base[rows]
                )
    return report


def main() -> None:
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["season", "game_type", "control_success"],
        low_memory=False,
    )
    active = raw["season"].isin([2023, 2024]).to_numpy()
    active_raw = raw.loc[active].reset_index(drop=True)
    season = active_raw["season"].to_numpy(np.int16)
    game_type = active_raw["game_type"].astype(str).to_numpy()

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = archive["blended"].astype(float)
        target = archive["target"].astype(float)
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = archive["blended"].astype(float)

    # Reconstruct the public-tested v56 step on the 2024 validation rows.
    base = v54.copy()
    rows_2024 = season == 2024
    f_2024 = rows_2024 & (game_type == "F")
    base[f_2024] = sigmoid(logit(v38[f_2024]) + F_SCALE * (
        logit(v54[f_2024]) - logit(v38[f_2024])
    ))

    centres = {
        year: float(raw.loc[raw["season"] < year, "control_success"].mean())
        for year in (2023, 2024)
    }
    production_centre = float(raw["control_success"].mean())
    regular = game_type == "R"

    candidates = []
    for transform in ("linear", "logit"):
        for alpha in np.round(np.arange(1.0, 1.101, 0.005), 3):
            prediction = base.copy()
            for year in (2023, 2024):
                rows = regular & (season == year)
                centre = centres[year]
                if transform == "linear":
                    prediction[rows] = np.clip(
                        centre + alpha * (base[rows] - centre), 1e-6, 1.0 - 1e-6
                    )
                else:
                    centre_logit = logit(np.array([centre]))[0]
                    prediction[rows] = sigmoid(
                        centre_logit + alpha * (logit(base[rows]) - centre_logit)
                    )
            gains = score_slices(target, base, prediction, season, game_type)
            year_gains = [gains[f"{year}/all"] for year in (2023, 2024)]
            r_gains = [gains[f"{year}/R"] for year in (2023, 2024)]
            quarter_gains = [
                gains[f"{year}/q{quarter}"]
                for year in (2023, 2024) for quarter in range(1, 5)
            ]
            candidates.append({
                "transform": transform,
                "alpha": float(alpha),
                "gains": gains,
                "min_year": float(min(year_gains)),
                "mean_year": float(np.mean(year_gains)),
                "min_r": float(min(r_gains)),
                "mean_r": float(np.mean(r_gains)),
                "min_quarter": float(min(quarter_gains)),
            })

    robust = sorted(
        candidates,
        key=lambda row: (row["min_year"], row["min_quarter"], row["mean_year"]),
        reverse=True,
    )
    report = {
        "method": "R-only affine sharpening around strictly-prior training mean",
        "fold_centres": centres,
        "production_centre": production_centre,
        "base_summary": {
            str(year): {
                "rows": int((season == year).sum()),
                "target_mean": float(target[season == year].mean()),
                "prediction_mean": float(base[season == year].mean()),
                "R_target_mean": float(target[(season == year) & regular].mean()),
                "R_prediction_mean": float(base[(season == year) & regular].mean()),
            }
            for year in (2023, 2024)
        },
        "robust": robust[:30],
        "all": candidates,
    }
    output = ROOT / "research/v57_affine_sharpening.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "centres": centres,
        "production_centre": production_centre,
        "base_summary": report["base_summary"],
        "top_robust": robust[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
