"""Promote the stable three-axis reference transfer over the v64 anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v66_reference_deviations import (
    TARGET, add_context, nested_deviation_table,
)
from research_v66_hierarchical_residual import pitcher_bootstrap, report_segments


ROOT = Path(__file__).resolve().parent
VERSION = "v66_reference_nested_deviations"
OVERALL_SCALE = 0.60
AXES = (
    ("platoon", "pitcher_id", "pitcher_hand_key", 300.0, 0.20),
    (
        "advantage", "pitcher_hand_key", "pitcher_hand_advantage_key",
        2000.0, 0.825,
    ),
    ("runner", "pitcher_hand_key", "runner_key", 2000.0, 0.45),
)
CLIP = (0.005, 0.995)


def serialize_axis(
    history: pd.DataFrame,
    name: str,
    parent: str,
    child: str,
    shrinkage: float,
    reference_weight: float,
) -> dict[str, object]:
    table = nested_deviation_table(history, parent, child, shrinkage)
    return {
        "name": name,
        "keys": table.index.astype(str).tolist(),
        "deltas": table.astype(float).tolist(),
        "weight": OVERALL_SCALE * reference_weight,
        "shrinkage": shrinkage,
        "unknown_key_delta": 0.0,
    }


def main() -> None:
    report = json.loads((
        ROOT / "research/v66_reference_deviations.json"
    ).read_text(encoding="utf-8"))
    if not report["strict_gate"]:
        raise ValueError("the reference-deviation research gate did not pass")
    with np.load(ROOT / "outputs/v64_oof_predictions.npz") as archive:
        v64 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v66_reference_deviations_oof.npz") as archive:
        research_oof = {key: archive[key] for key in archive.files}
    target = v64["target"].astype(float)
    season = v64["season"].astype(int)
    anchor = v64["blended"].astype(float)
    components = research_oof["components"].astype(float)
    correction = OVERALL_SCALE * (
        0.20 * components[:, 0]
        + 0.825 * components[:, 1]
        + 0.45 * components[:, 3]
    )
    prediction = np.clip(anchor + correction, *CLIP)

    columns = [
        "season", "game_type", "pitcher_id", "batter_hand",
        "balls_before", "strikes_before", "num_runners_on", TARGET,
    ]
    raw = add_context(pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    ))
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    validation = {
        str(year): report_segments(
            target[season == year], anchor[season == year],
            prediction[season == year], regular[season == year],
        ) for year in (2023, 2024)
    }
    bootstrap = {
        str(year): pitcher_bootstrap(
            target[season == year], anchor[season == year],
            prediction[season == year],
            rows.loc[season == year, "pitcher_id"].to_numpy(),
            100000, 674000 + year,
        ) for year in (2023, 2024)
    }
    if not all(
        validation[str(year)]["gain"] > 0
        and min(validation[str(year)]["half_gains"]) > 0
        and min(validation[str(year)]["quarter_gains"]) > 0
        and validation[str(year)]["regular_gain"] > 0
        and validation[str(year)]["futures_gain"] > 0
        for year in (2023, 2024)
    ):
        raise ValueError("v66 stability gate failed")

    production_axes = [
        serialize_axis(raw, *specification) for specification in AXES
    ]
    np.savez_compressed(
        ROOT / "outputs/v66_oof_predictions.npz",
        **{key: value for key, value in v64.items() if key != "blended"},
        blended=prediction,
    )
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "v64_public_method_transfer" not in metadata.get("model_names", []):
        raise ValueError("v66 requires a complete v64 bundle")
    rejected = {
        "v62_public_residual_frontier", "v63_train_trend_calibration",
        "v65_prediction_gap_meta", VERSION,
    }
    metadata["model_names"] = [
        name for name in metadata["model_names"] if name not in rejected
    ] + [VERSION]
    for name in rejected - {VERSION}:
        metadata.pop(name, None)
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "baseline": "v64_public_method_transfer",
        "axes": production_axes,
        "source_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "overall_scale": OVERALL_SCALE,
        "validation": validation,
        "pitcher_bootstrap": bootstrap,
        "row_independent_inference": True,
        "external_table_model_or_prediction_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
        "v62_or_v63_component_used": False,
        "projected_public_range": [1135.0, 1144.0],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "baseline_public_score": 1135.1,
        "axes": [axis["name"] for axis in production_axes],
        "axis_weights": {
            axis["name"]: axis["weight"] for axis in production_axes
        },
        "validation": validation,
        "pitcher_bootstrap": bootstrap,
        "projected_public_range": [1135.0, 1144.0],
        "rules": metadata[VERSION] | {"axes": None, "validation": None,
                                     "pitcher_bootstrap": None},
    }
    (ROOT / "research/v66_promotion.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(promotion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
