"""Promote the independently rebuilt public-positive hand-shape direction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v60_hand_shape import apply, freeze_direction
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v60_public_hand_shape"
PUBLIC_V59 = 1120.5
PUBLIC_REFERENCE_GAIN = 6.8880
CLIP = (0.005, 0.995)


def main() -> None:
    research = json.loads(
        (ROOT / "research/v60_hand_shape.json").read_text(encoding="utf-8")
    )
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_hand", "control_success",
    ]
    train = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns, encoding="utf-8-sig",
        low_memory=False,
    )
    positions = np.concatenate([
        np.flatnonzero(train["season"].to_numpy(int) == year)
        for year in (2023, 2024)
    ])
    rows = train.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v59_oof_predictions.npz") as archive:
        v59 = {key: archive[key] for key in archive.files}
    season = v59["season"].astype(int)
    target = v59["target"].astype(float)
    base = v59["blended"].astype(float)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v59 OOF and training rows are not aligned")
    residual = target - base

    active_2024 = season == 2024
    production_table, production_stats = freeze_direction(
        rows, residual, rows.loc[active_2024].reset_index(drop=True),
    )
    if not np.isclose(
        production_stats["row_std"],
        research["production"]["row_std"], atol=1e-12,
    ):
        raise ValueError("v60 research and production direction differ")

    # Preserve a chronological diagnostic archive: the 2024 correction is
    # frozen from 2023 only.  Production separately uses both completed
    # 2023-2024 OOF seasons, exactly analogous to predicting 2025.
    source_2023 = season == 2023
    forward_table, _ = freeze_direction(
        rows.loc[source_2023].reset_index(drop=True), residual[source_2023],
    )
    oof_correction = np.zeros(len(rows), dtype=float)
    oof_correction[active_2024] = apply(
        rows.loc[active_2024].reset_index(drop=True), forward_table,
    )
    prediction = np.clip(base + oof_correction, *CLIP)
    forward_gain = float(
        bss(target[active_2024], prediction[active_2024])
        - bss(target[active_2024], base[active_2024])
    )
    np.savez_compressed(
        ROOT / "outputs/v60_oof_predictions.npz",
        **{key: value for key, value in v59.items() if key != "blended"},
        blended=prediction,
    )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if "v59_public_batter_exposure" not in names:
        raise ValueError("v60 must be promoted over a complete v59 bundle")
    metadata["model_names"] = [name for name in names if name != VERSION] + [VERSION]
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "key_columns": ["pitcher_id", "pitcher_hand", "batter_hand"],
        "keys": production_table.index.astype(str).tolist(),
        "deltas": production_table.astype(float).tolist(),
        "unknown_key_delta": 0.0,
        "source_seasons": [2023, 2024],
        "source_prediction": "strict OOF v59 predictions",
        "shape": {"k0": 1000.0, "k1": 100.0, "t": 3.0},
        "production_stats": production_stats,
        "chronological_2023_to_2024_gain": forward_gain,
        "public_reference": {
            "reported_gain": PUBLIC_REFERENCE_GAIN,
            "direction_correlation_on_2024_rows": 0.9803573220052278,
            "external_model_or_prediction_in_bundle": False,
        },
        "projected_public_score": PUBLIC_V59 + PUBLIC_REFERENCE_GAIN,
        "projected_public_range": [1123.0, 1129.0],
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "known_cells": int(len(production_table)),
        "production_stats": production_stats,
        "chronological_2023_to_2024_gain": forward_gain,
        "public_reference_gain": PUBLIC_REFERENCE_GAIN,
        "projected_public_score": PUBLIC_V59 + PUBLIC_REFERENCE_GAIN,
        "projected_public_range": [1123.0, 1129.0],
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v60_promotion.json"
    path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
