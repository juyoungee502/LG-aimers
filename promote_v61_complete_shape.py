"""Promote the independently rebuilt batter-shape and pitcher-log tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v61_complete_shape import (
    CLIP,
    expected_gain,
    freeze_batter_shape,
    freeze_pitcher_log,
    row_values,
)
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v61_public_complete_shape"


def serialized_table(column: str, table: pd.Series) -> dict:
    return {
        "id_column": column,
        "keys": [str(value) for value in table.index.tolist()],
        "deltas": table.astype(float).tolist(),
        "unknown_player_delta": 0.0,
    }


def main() -> None:
    research = json.loads(
        (ROOT / "research/v61_complete_shape.json").read_text(encoding="utf-8")
    )
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_id", "batter_hand", "control_success",
    ]
    train = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    positions = np.concatenate([
        np.flatnonzero(train["season"].to_numpy(int) == year)
        for year in (2023, 2024)
    ])
    rows = train.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v60_oof_predictions.npz") as archive:
        v60 = {key: archive[key] for key in archive.files}
    season = v60["season"].astype(int)
    target = v60["target"].astype(float)
    base = v60["blended"].astype(float)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v60 OOF and training rows are not aligned")
    residual = target - base
    active_2024 = season == 2024
    reference = rows.loc[active_2024].reset_index(drop=True)

    batter_table, batter_stats = freeze_batter_shape(rows, residual, reference)
    pitcher_table, pitcher_stats = freeze_pitcher_log(rows, residual, reference)
    if not np.isclose(
        batter_stats["row_std"],
        research["production"]["batter_shape"]["row_std"], atol=1e-12,
    ) or not np.isclose(
        pitcher_stats["row_std"],
        research["production"]["pitcher_log"]["row_std"], atol=1e-12,
    ):
        raise ValueError("v61 research and production directions differ")

    source_2023 = season == 2023
    source_rows = rows.loc[source_2023].reset_index(drop=True)
    batter_forward, _ = freeze_batter_shape(source_rows, residual[source_2023])
    pitcher_forward, _ = freeze_pitcher_log(source_rows, residual[source_2023])
    oof_correction = np.zeros(len(rows), dtype=float)
    validation_rows = rows.loc[active_2024].reset_index(drop=True)
    oof_correction[active_2024] = (
        row_values(validation_rows, "batter_id", batter_forward)
        + row_values(validation_rows, "pitcher_id", pitcher_forward)
    )
    prediction = np.clip(base + oof_correction, *CLIP)
    forward_gain = float(
        bss(target[active_2024], prediction[active_2024])
        - bss(target[active_2024], base[active_2024])
    )
    np.savez_compressed(
        ROOT / "outputs/v61_oof_predictions.npz",
        **{key: value for key, value in v60.items() if key != "blended"},
        blended=prediction,
    )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if "v60_public_hand_shape" not in names:
        raise ValueError("v61 must be promoted over a complete v60 bundle")
    metadata["model_names"] = [name for name in names if name != VERSION] + [VERSION]
    metadata["version"] = VERSION
    projection = expected_gain()
    metadata[VERSION] = {
        "source_seasons": [2023, 2024],
        "source_prediction": "strict OOF v60 predictions",
        "batter_shape": {
            **serialized_table("batter_id", batter_table),
            "shape": {"k0": 20000.0, "k1": 2000.0, "t": -4.5, "strength": 0.85},
            "production_stats": batter_stats,
            "public_direction_correlation": 0.9063809167499408,
        },
        "pitcher_log": {
            **serialized_table("pitcher_id", pitcher_table),
            "shape": {"k0": 50000.0, "s": 0.4, "strength": 0.8},
            "production_stats": pitcher_stats,
            "public_direction_correlation": 0.999560094851847,
        },
        "chronological_2023_to_2024_gain": forward_gain,
        "projected_component_gains": projection,
        "projected_public_score": projection["score"],
        "projected_public_range": [1131.0, 1138.0],
        "row_independent_inference": True,
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "batter_cells": int(len(batter_table)),
        "pitcher_cells": int(len(pitcher_table)),
        "chronological_2023_to_2024_gain": forward_gain,
        "projection": projection,
        "projected_public_range": [1131.0, 1138.0],
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v61_promotion.json"
    path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
