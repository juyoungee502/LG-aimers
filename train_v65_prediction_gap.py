"""Fit and promote the conservative v65 prediction-gap meta correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research_v65_prediction_gap_meta import (
    centered_residual, fit_predict, load_meta,
)
from train_v25_temporal_portfolio import bss
from v65_prediction_gap import CLIP, prediction_gap_correction


ROOT = Path(__file__).resolve().parent
VERSION = "v65_prediction_gap_meta"


def score_gain(target: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(target, np.clip(candidate, *CLIP)) - bss(target, base))


def main() -> None:
    report_path = ROOT / "research/v65_prediction_gap_meta.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("strict_gate") or report.get("selected") is None:
        raise RuntimeError("v65 research did not pass the strict promotion gate")
    selected = report["selected"]
    alpha = float(selected["ridge"])
    r_scale = float(selected["r_scale"])
    f_scale = float(selected["f_scale"])

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        oof = {key: archive[key] for key in archive.files}
    target = oof["target"].astype(np.float64)
    anchor = oof["blended"].astype(np.float64)
    season = oof["season"].astype(int)
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "control_success",
            "asof_pitcher_success_rate", "asof_batter_success_rate",
            "asof_pitcher_n", "asof_batter_n", "balls_before", "strikes_before",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(target) or not np.array_equal(
        rows["control_success"].to_numpy(float), target,
    ):
        raise ValueError("v64 OOF rows are not aligned with train.csv")

    features, lineage = load_meta(anchor, target, season, rows)
    if lineage != report["oof_lineage"]:
        raise ValueError("v65 research lineage changed before promotion")
    game = rows["game_type"].astype(str).to_numpy()
    positions23 = np.flatnonzero(season == 2023)
    positions24 = np.flatnonzero(season == 2024)
    first23, second23 = np.array_split(positions23, 2)

    # Honest archived predictions: the 2023 second half sees only its first
    # half, and 2024 sees only 2023.  The untouched first half is retained.
    oof_correction = np.zeros(len(anchor), dtype=np.float64)
    validation_folds = (
        ("2023_second_half", first23, second23),
        ("2024_forward", positions23, positions24),
    )
    fold_audits = {}
    for name, fit_index, valid_index in validation_folds:
        residual = centered_residual(
            target[fit_index], anchor[fit_index], game[fit_index],
        )
        raw_correction = fit_predict(
            features, fit_index, valid_index, residual, alpha,
        )
        scales = np.where(game[valid_index] == "R", r_scale, f_scale)
        oof_correction[valid_index] = scales * raw_correction
        candidate = np.clip(
            anchor[valid_index] + oof_correction[valid_index], *CLIP,
        )
        fold_audits[name] = {
            "gain": score_gain(target[valid_index], anchor[valid_index], candidate),
            "mean_absolute_change": float(np.mean(np.abs(
                candidate - anchor[valid_index]
            ))),
        }
    corrected = np.clip(anchor + oof_correction, *CLIP)

    # Freeze a final direction using both OOF seasons.  The validation-selected
    # regularisation and tiny R/F scales remain unchanged.
    residual = centered_residual(target, anchor, game)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(standardized, residual)
    configuration = {
        "baseline": "v64_public_method_transfer",
        "method": "standardized_ridge_prediction_gap_residual",
        "ridge_alpha": alpha,
        "fit_intercept": False,
        "r_scale": r_scale,
        "f_scale": f_scale,
        "game_type_regular": "R",
        "stage_names": lineage,
        "member_names": oof["model_names"].astype(str).tolist(),
        "feature_columns": list(features.columns),
        "feature_mean": scaler.mean_.astype(float).tolist(),
        "feature_scale": scaler.scale_.astype(float).tolist(),
        "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        "training_seasons": [2023, 2024],
        "target_group_centering": ["R", "F"],
        "validation": {
            "folds": fold_audits,
            "bootstrap": report["bootstrap"],
            "strict_gate": True,
        },
        "row_independent_inference": True,
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
        "v62_or_v63_component_used": False,
    }

    # One exact implementation-parity assertion before touching metadata.
    stage_predictions = {}
    for version in lineage:
        with np.load(ROOT / f"outputs/{version}_oof_predictions.npz") as archive:
            stage_predictions[version] = archive["blended"].astype(float)
    member_predictions = {
        name: oof["predictions"][:, index].astype(float)
        for index, name in enumerate(configuration["member_names"])
    }
    frozen_correction = prediction_gap_correction(
        rows, anchor, member_predictions, stage_predictions,
        oof["base_blended"].astype(float),
        oof["trackman_context"].astype(float),
        oof["f_specialist"].astype(float), configuration,
    )
    direct_correction = np.where(
        game == "R", r_scale, f_scale,
    ) * model.predict(standardized)
    if not np.allclose(frozen_correction, direct_correction, atol=1e-12, rtol=1e-10):
        raise RuntimeError("v65 frozen inference differs from fitted correction")

    np.savez_compressed(
        ROOT / "outputs/v65_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=corrected,
        prediction_gap_correction=oof_correction,
    )
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v64_public_method_transfer":
        raise ValueError(f"v65 requires v64 metadata, found {metadata.get('version')}")
    rejected = {
        "v62_public_residual_frontier", "v63_train_trend_calibration", VERSION,
    }
    names = [name for name in metadata["model_names"] if name not in rejected]
    metadata["model_names"] = names + [VERSION]
    metadata.pop("v62_public_residual_frontier", None)
    metadata.pop("v63_train_trend_calibration", None)
    metadata["version"] = VERSION
    metadata[VERSION] = configuration
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )

    promotion = {
        "version": VERSION,
        "baseline": "v64_public_method_transfer",
        "selected": selected,
        "folds": fold_audits,
        "bootstrap": report["bootstrap"],
        "feature_count": len(configuration["feature_columns"]),
        "lineage": lineage,
        "production_mean_absolute_correction": float(np.mean(np.abs(
            frozen_correction
        ))),
        "projected_public_range": [1135.0, 1142.0],
        "rules": report["rules"],
    }
    path = ROOT / "research/v65_promotion.json"
    path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(promotion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
