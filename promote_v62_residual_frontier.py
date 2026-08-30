"""Promote the v62 residual frontier over the complete v61 bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v60_hand_shape import freeze_direction as freeze_hand_shape
from research_v62_residual_frontier import (
    CLIP,
    V61_SCALE_DELTA,
    apply_table,
    freeze_c4n_mirror,
    freeze_d0_shape,
    freeze_hd,
)
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v62_public_residual_frontier"


def serialize(columns: list[str], table: pd.Series) -> dict:
    return {
        "key_columns": columns,
        "keys": table.index.astype(str).tolist(),
        "deltas": table.astype(float).tolist(),
        "unknown_key_delta": 0.0,
    }


def main() -> None:
    research = json.loads(
        (ROOT / "research/v62_residual_frontier.json").read_text(encoding="utf-8")
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
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        v61 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v60_oof_predictions.npz") as archive:
        base_v60 = archive["blended"].astype(float)
    season = v61["season"].astype(int)
    target = v61["target"].astype(float)
    base = v61["blended"].astype(float)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v61 OOF and training rows are not aligned")
    residual = target - base
    active_2024 = season == 2024
    reference = rows.loc[active_2024].reset_index(drop=True)

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if "v61_public_complete_shape" not in names:
        raise ValueError("v62 must be promoted over a complete v61 bundle")
    hand_config = metadata["v60_public_hand_shape"]
    hand_table = pd.Series(
        dict(zip(hand_config["keys"], hand_config["deltas"])), dtype=float,
    )
    c4n, c4n_stats = freeze_c4n_mirror(rows, hand_table, reference)
    hd, hd_stats = freeze_hd(rows, residual, reference)
    d0, d0_stats = freeze_d0_shape(rows, residual, reference)
    for name, stats in (("c4n", c4n_stats), ("hd", hd_stats), ("d0", d0_stats)):
        expected = research["production"][name]["row_std"]
        if not np.isclose(stats["row_std"], expected, atol=1e-12):
            raise ValueError(f"v62 {name} research and production directions differ")

    source_2023 = season == 2023
    source_rows = rows.loc[source_2023].reset_index(drop=True)
    source_residual = residual[source_2023]
    source_hand, _ = freeze_hand_shape(source_rows, source_residual, source_rows)
    c4n_forward, _ = freeze_c4n_mirror(source_rows, source_hand, source_rows)
    hd_forward, _ = freeze_hd(source_rows, source_residual, source_rows)
    d0_forward, _ = freeze_d0_shape(source_rows, source_residual, source_rows)
    validation_rows = rows.loc[active_2024].reset_index(drop=True)
    oof_correction = V61_SCALE_DELTA * (base - base_v60)
    oof_correction[active_2024] += (
        apply_table(validation_rows, ["pitcher_id", "pitcher_hand", "batter_hand"], c4n_forward)
        + apply_table(validation_rows, ["pitcher_id", "pitcher_hand", "batter_hand"], hd_forward)
        + apply_table(validation_rows, ["pitcher_id", "batter_hand"], d0_forward)
    )
    prediction = np.clip(base + oof_correction, *CLIP)
    forward_gain = float(
        bss(target[active_2024], prediction[active_2024])
        - bss(target[active_2024], base[active_2024])
    )
    np.savez_compressed(
        ROOT / "outputs/v62_oof_predictions.npz",
        **{key: value for key, value in v61.items() if key != "blended"},
        blended=prediction,
    )

    metadata["model_names"] = [name for name in names if name != VERSION] + [VERSION]
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "source_seasons": [2023, 2024],
        "source_prediction": "strict OOF v61 predictions",
        "v61_component_scale_delta": V61_SCALE_DELTA,
        "c4n_mirror": {
            **serialize(["pitcher_id", "pitcher_hand", "batter_hand"], c4n),
            "production_stats": c4n_stats,
            "public_direction_correlation": 0.9997962691489939,
        },
        "residual_hand": {
            **serialize(["pitcher_id", "pitcher_hand", "batter_hand"], hd),
            "production_stats": hd_stats,
            "public_direction_correlation": 0.9723600795906776,
        },
        "d0_shape": {
            **serialize(["pitcher_id", "batter_hand"], d0),
            "production_stats": d0_stats,
            "public_direction_correlation": 0.7094279283165271,
        },
        "chronological_2023_to_2024_gain": forward_gain,
        "projection": research["projection"],
        "projected_public_score": research["projection"]["score"],
        "projected_public_range": [1135.0, 1144.0],
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
        "c4n_cells": int(len(c4n)), "hd_cells": int(len(hd)),
        "d0_cells": int(len(d0)),
        "chronological_2023_to_2024_gain": forward_gain,
        "projection": research["projection"],
        "projected_public_range": [1135.0, 1144.0],
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v62_promotion.json"
    path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
