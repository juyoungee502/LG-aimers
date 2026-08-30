"""Audit the deterministic high-usage player direction for v59.

A contemporary public experiment on the same competition found that a frozen,
label-free batter exposure direction improved its public score materially.  We
rebuild that idea independently: count each player's rows in the two completed
seasons before the prediction season, center counts over known players, then do
one row-local ID lookup.  No external model or external prediction is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
# The public result's reported deployed probability slope at its near-optimal
# s=0.4 point, reconstructed from its documented normalization.
REFERENCE_SLOPE = 2.0907659421884613e-6
CLIP = (0.005, 0.995)


def frozen_count_direction(
    all_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_season: int,
    id_column: str,
    window: int,
) -> tuple[np.ndarray, dict]:
    source = all_rows[all_rows["season"].between(
        validation_season - window, validation_season - 1,
    )]
    counts = source.groupby(id_column, sort=False).size().astype(float)
    center = float(counts.mean())
    lookup = counts - center
    direction = validation_rows[id_column].map(lookup).fillna(0.0).to_numpy(float)
    return direction, {
        "source_seasons": [validation_season - window, validation_season - 1],
        "known_players": int(len(counts)),
        "entity_mean_count": center,
        "row_mean_direction": float(direction.mean()),
        "row_std_direction": float(direction.std()),
        "unknown_row_fraction": float(validation_rows[id_column].map(counts).isna().mean()),
    }


def score_segments(rows, target, base, correction) -> dict[str, float]:
    month = rows["game_month"].to_numpy(int)
    game_type = rows["game_type"].astype(str).to_numpy()
    masks = {
        "all": np.ones(len(rows), bool), "H1": month <= 6, "H2": month >= 7,
        "R": game_type == "R", "F": game_type == "F",
    }
    candidate = np.clip(base + correction, *CLIP)
    return {
        name: float(bss(target[mask], candidate[mask]) - bss(target[mask], base[mask]))
        for name, mask in masks.items() if mask.any()
    }


def main() -> None:
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "control_success",
    ]
    all_rows = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns, encoding="utf-8-sig", low_memory=False,
    )
    positions = np.concatenate([
        np.flatnonzero(all_rows["season"].to_numpy(int) == year)
        for year in (2023, 2024)
    ])
    rows = all_rows.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v58_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        seasons = archive["season"].astype(int)
    if len(rows) != len(target) or not np.array_equal(rows["season"].to_numpy(int), seasons):
        raise ValueError("training rows and v58 OOF are not aligned")

    multipliers = (-0.5, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    report = {"baseline": "v58_public_feedback_counterstep", "candidates": {}}
    for role in ("batter", "pitcher"):
        for window in (1, 2, 3):
            name = f"{role}_{window}season_count"
            entry = {}
            for year in (2023, 2024):
                mask = seasons == year
                validation = rows.loc[mask].reset_index(drop=True)
                raw_direction, stats = frozen_count_direction(
                    all_rows, validation, year, f"{role}_id", window,
                )
                entry[str(year)] = {
                    "stats": stats,
                    "gains": {
                        str(multiplier): score_segments(
                            validation, target[mask], base[mask],
                            multiplier * REFERENCE_SLOPE * raw_direction,
                        )
                        for multiplier in multipliers
                    },
                }
            report["candidates"][name] = entry

    ranking = []
    for name, entry in report["candidates"].items():
        for multiplier in multipliers:
            gains = {
                year: entry[year]["gains"][str(multiplier)]["all"]
                for year in ("2023", "2024")
            }
            ranking.append({
                "name": name, "reference_multiplier": multiplier,
                "year_gains": gains, "min_year_gain": min(gains.values()),
                "mean_year_gain": float(np.mean(list(gains.values()))),
                "segments_2024": entry["2024"]["gains"][str(multiplier)],
            })
    ranking.sort(key=lambda item: (item["min_year_gain"], item["mean_year_gain"]), reverse=True)
    report.update({
        "reference_probability_slope": REFERENCE_SLOPE,
        "top_ranked": ranking[:30],
        "external_models_or_predictions_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    })
    path = ROOT / "research/v59_public_count_direction.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["top_ranked"][:20], indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
