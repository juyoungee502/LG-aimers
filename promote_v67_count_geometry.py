"""Promote the audited v65 + original low-rank count geometry as v67."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v67_count_geometry import (
    TARGET, apply_geometry, build_geometry, count_state,
)
from v67_count_geometry import apply_count_geometry


ROOT = Path(__file__).resolve().parent
VERSION = "v67_original_count_geometry"


def main() -> None:
    report = json.loads(
        (ROOT / "research/v67_count_geometry.json").read_text(encoding="utf-8")
    )
    if not report.get("strict_gate") or report.get("selected") is None:
        raise RuntimeError("v67 count geometry did not pass the strict gate")
    selected = report["selected"]
    decay = float(selected["decay"])
    shrinkage = float(selected["shrinkage"])
    rank = int(selected["rank"])
    scale = float(selected["scale"])

    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "balls_before",
            "strikes_before", TARGET,
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    raw["count_code"] = count_state(raw)
    with np.load(ROOT / "outputs/v65_oof_predictions.npz", allow_pickle=True) as archive:
        oof = {key: archive[key] for key in archive.files}
    target = oof["target"].astype(np.float64)
    season = oof["season"].astype(int)
    anchor = oof["blended"].astype(np.float64)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(target) or not np.array_equal(
        rows[TARGET].to_numpy(float), target,
    ):
        raise ValueError("v65 OOF rows do not align with train.csv")

    oof_correction = np.zeros(len(anchor), dtype=np.float64)
    fold_audit = {}
    for year in (2023, 2024):
        geometry = build_geometry(
            raw.loc[raw["season"].lt(year)], year,
            decay=decay, shrinkage=shrinkage, rank=rank,
        )
        validation = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        direction, coverage = apply_geometry(validation, geometry)
        mask = season == year
        oof_correction[mask] = scale * direction
        fold_audit[str(year)] = {
            "coverage": coverage,
            "mean_absolute_correction": float(np.mean(np.abs(scale * direction))),
            **selected["reports"][str(year)],
            "bootstrap": report["bootstrap"][str(year)],
        }
    corrected = np.clip(anchor + oof_correction, 0.005, 0.995)
    research_prediction = np.load(
        ROOT / "outputs/v67_count_geometry_oof.npz", allow_pickle=False,
    )["blended"].astype(np.float64)
    if not np.allclose(corrected, research_prediction, atol=1e-12, rtol=1e-10):
        raise RuntimeError("v67 promotion differs from audited OOF prediction")

    # Production tables see all official 2019--2024 train seasons and remain
    # completely frozen while 2025 evaluation rows are scored independently.
    production = build_geometry(
        raw.loc[raw["season"].lt(2025)], 2025,
        decay=decay, shrinkage=shrinkage, rank=rank,
    )
    configuration = {
        "baseline": "v65_prediction_gap_meta",
        "method": "exposure_shrunk_rank1_pitcher_by_count_interaction",
        "prediction_year": 2025,
        "history_seasons": sorted(
            int(value) for value in raw.loc[raw["season"].lt(2025), "season"].unique()
        ),
        "decay": decay,
        "shrinkage": shrinkage,
        "rank": rank,
        "scale": scale,
        "pitcher_ids": [str(value) for value in production["pitcher_ids"]],
        "values": np.asarray(production["values"], dtype=float).tolist(),
        "validation": {
            "folds": fold_audit,
            "strict_gate": True,
            "selection_policy": report["selection_policy"],
        },
        "row_independent_inference": True,
        "external_reference_model_or_prediction_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
        "v62_v63_or_v66_component_used": False,
    }
    parity = apply_count_geometry(raw.tail(500).reset_index(drop=True), configuration)
    direct, _ = apply_geometry(raw.tail(500).reset_index(drop=True), production)
    if not np.allclose(parity, scale * direct, atol=1e-12, rtol=1e-10):
        raise RuntimeError("v67 frozen lookup differs from production geometry")

    np.savez_compressed(
        ROOT / "outputs/v67_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=corrected,
        count_geometry_correction=oof_correction,
    )
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v65_prediction_gap_meta":
        raise ValueError(
            f"v67 requires v65 metadata, found {metadata.get('version')}"
        )
    rejected = {
        "v62_public_residual_frontier", "v63_train_trend_calibration",
        "v66_reference_nested_deviations", VERSION,
    }
    metadata["model_names"] = [
        name for name in metadata["model_names"] if name not in rejected
    ] + [VERSION]
    for name in rejected:
        metadata.pop(name, None)
    metadata["version"] = VERSION
    metadata[VERSION] = configuration
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )

    promotion = {
        "version": VERSION,
        "baseline": "v65_prediction_gap_meta_public_1135_0",
        "selected": {
            key: selected[key]
            for key in ("decay", "shrinkage", "rank", "scale")
        },
        "folds": fold_audit,
        "production_pitcher_coverage_table_size": len(production["pitcher_ids"]),
        "production_mean_absolute_correction": float(np.mean(np.abs(parity))),
        "projected_public_range": [1135.0, 1143.0],
        "rules": report["rules"],
    }
    path = ROOT / "research/v67_promotion.json"
    path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(promotion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
