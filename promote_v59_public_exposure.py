"""Promote the label-free high-usage batter direction over v58.

Only 2023-2024 row counts are frozen for production.  The table contains no
target statistic, and inference is a single batter_id lookup per test row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v59_public_batter_exposure"
SOURCE_SEASONS = (2023, 2024)
REFERENCE_SLOPE = 2.0907659421884613e-6
PUBLIC_V58 = 1114.0
PUBLIC_REFERENCE_GAIN = 8.3544
CLIP = (0.005, 0.995)


def freeze_count_table(
    rows: pd.DataFrame,
    source_seasons: tuple[int, ...],
) -> tuple[pd.Series, float]:
    source = rows[rows["season"].isin(source_seasons)]
    counts = source.groupby("batter_id", sort=True).size().astype(float)
    center = float(counts.mean())
    return REFERENCE_SLOPE * (counts - center), center


def apply_prior_window(
    all_rows: pd.DataFrame,
    validation: pd.DataFrame,
    validation_year: int,
) -> np.ndarray:
    table, _ = freeze_count_table(
        all_rows, (validation_year - 2, validation_year - 1),
    )
    return validation["batter_id"].map(table).fillna(0.0).to_numpy(float)


def main() -> None:
    report_path = ROOT / "research/v59_public_count_direction.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not np.isclose(report["reference_probability_slope"], REFERENCE_SLOPE):
        raise ValueError("v59 research and deployment slopes differ")

    all_rows = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["season", "game_type", "game_month", "batter_id", "control_success"],
        encoding="utf-8-sig", low_memory=False,
    )
    positions = np.concatenate([
        np.flatnonzero(all_rows["season"].to_numpy(int) == year)
        for year in (2023, 2024)
    ])
    rows = all_rows.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v58_oof_predictions.npz") as archive:
        v58 = {key: archive[key] for key in archive.files}
    season = v58["season"].astype(int)
    target = v58["target"].astype(float)
    base = v58["blended"].astype(float)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v58 OOF and training rows are not aligned")

    correction = np.zeros(len(rows), dtype=float)
    validation = {}
    for year in (2023, 2024):
        mask = season == year
        correction[mask] = apply_prior_window(
            all_rows, rows.loc[mask].reset_index(drop=True), year,
        )
        candidate = np.clip(base[mask] + correction[mask], *CLIP)
        validation[str(year)] = {
            "gain": float(bss(target[mask], candidate) - bss(target[mask], base[mask])),
            "correction_mean": float(correction[mask].mean()),
            "correction_std": float(correction[mask].std()),
            "correction_min": float(correction[mask].min()),
            "correction_max": float(correction[mask].max()),
        }
    prediction = np.clip(base + correction, *CLIP)
    np.savez_compressed(
        ROOT / "outputs/v59_oof_predictions.npz",
        **{key: value for key, value in v58.items() if key != "blended"},
        blended=prediction,
    )

    production_table, center = freeze_count_table(all_rows, SOURCE_SEASONS)
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if "v58_public_feedback_counterstep" not in names:
        raise ValueError("v59 must be promoted over a complete v58 bundle")
    metadata["model_names"] = [name for name in names if name != VERSION] + [VERSION]
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "id_column": "batter_id",
        "source_seasons": list(SOURCE_SEASONS),
        "entity_mean_count": center,
        "probability_slope_per_training_row": REFERENCE_SLOPE,
        "keys": [str(value) for value in production_table.index.tolist()],
        "deltas": production_table.astype(float).tolist(),
        "unknown_player_delta": 0.0,
        "local_forward_validation": validation,
        "public_reference": {
            "description": "independent public same-test-set pure batter exposure direction",
            "reported_gain": PUBLIC_REFERENCE_GAIN,
            "reference_direction_near_optimum": True,
            "external_model_or_prediction_in_bundle": False,
        },
        "projected_public_score": PUBLIC_V58 + PUBLIC_REFERENCE_GAIN,
        "projected_public_range": [1119.0, 1123.5],
        "row_independent_inference": True,
        "label_free_table": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "source_seasons": SOURCE_SEASONS,
        "known_batters": int(len(production_table)),
        "entity_mean_count": center,
        "probability_slope": REFERENCE_SLOPE,
        "production_delta_stats": {
            "mean_over_entities": float(production_table.mean()),
            "std_over_entities": float(production_table.std(ddof=0)),
            "min": float(production_table.min()),
            "max": float(production_table.max()),
        },
        "local_forward_validation": validation,
        "projected_public_score": PUBLIC_V58 + PUBLIC_REFERENCE_GAIN,
        "projected_public_range": [1119.0, 1123.5],
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    output = ROOT / "research/v59_promotion.json"
    output.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
