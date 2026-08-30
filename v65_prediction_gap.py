"""Small, frozen prediction-gap correction used by v65.

The correction consumes only row-local values and predictions already produced
by this submission.  It never aggregates evaluation rows or reads outcomes at
inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CLIP = (0.005, 0.995)


def build_prediction_gap_features(
    rows: pd.DataFrame,
    anchor: np.ndarray,
    member_predictions: dict[str, np.ndarray],
    stage_predictions: dict[str, np.ndarray],
    base_blended: np.ndarray,
    trackman_prediction: np.ndarray,
    f_specialist_prediction: np.ndarray,
    *,
    member_names: list[str],
    stage_names: list[str],
) -> pd.DataFrame:
    """Recreate the exact meta-feature schema used by the forward audit."""
    anchor = np.asarray(anchor, dtype=np.float64)
    data: dict[str, np.ndarray] = {}
    for name in stage_names:
        data[f"gap_{name}"] = np.asarray(
            stage_predictions[name], dtype=np.float64,
        ) - anchor
    for index, name in enumerate(member_names):
        data[f"member_gap_{index}"] = np.asarray(
            member_predictions[name], dtype=np.float64,
        ) - anchor
    data["gap_base_blended"] = np.asarray(
        base_blended, dtype=np.float64,
    ) - anchor
    data["gap_trackman_context"] = np.asarray(
        trackman_prediction, dtype=np.float64,
    ) - anchor
    # The archived F specialist exists only for 2024 F rows.  It therefore has
    # no value in either forward-training source (2023 or 2023-H1).  Keep it and
    # the legacy aggregate slots inert so a full 2023+2024 refit cannot learn a
    # signal that was absent from both strict validation fits.
    del f_specialist_prediction
    data["gap_f_specialist"] = np.zeros(len(anchor), dtype=np.float64)
    data["gap_mean"] = np.zeros(len(anchor), dtype=np.float64)
    data["gap_std"] = np.zeros(len(anchor), dtype=np.float64)
    data["gap_min"] = np.zeros(len(anchor), dtype=np.float64)
    data["gap_max"] = np.zeros(len(anchor), dtype=np.float64)
    data["anchor_centered"] = anchor - 0.5

    out = pd.DataFrame(data)
    pitcher = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce",
    )
    batter = pd.to_numeric(
        rows["asof_batter_success_rate"], errors="coerce",
    )
    out["batter_pitcher_form_gap"] = (pitcher - batter).fillna(0.0).to_numpy()
    out["pitcher_log_n"] = np.log1p(pd.to_numeric(
        rows["asof_pitcher_n"], errors="coerce",
    ).fillna(0.0).clip(lower=0.0)).to_numpy()
    out["batter_log_n"] = np.log1p(pd.to_numeric(
        rows["asof_batter_n"], errors="coerce",
    ).fillna(0.0).clip(lower=0.0)).to_numpy()
    out["game_type_f"] = rows["game_type"].astype(str).eq("F").to_numpy(float)
    out["two_strike"] = pd.to_numeric(
        rows["strikes_before"], errors="coerce",
    ).eq(2).to_numpy(float)
    out["three_ball"] = pd.to_numeric(
        rows["balls_before"], errors="coerce",
    ).eq(3).to_numpy(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prediction_gap_correction(
    rows: pd.DataFrame,
    anchor: np.ndarray,
    member_predictions: dict[str, np.ndarray],
    stage_predictions: dict[str, np.ndarray],
    base_blended: np.ndarray,
    trackman_prediction: np.ndarray,
    f_specialist_prediction: np.ndarray,
    configuration: dict,
) -> np.ndarray:
    """Apply the frozen standardized Ridge direction with regime-safe scales."""
    features = build_prediction_gap_features(
        rows, anchor, member_predictions, stage_predictions, base_blended,
        trackman_prediction, f_specialist_prediction,
        member_names=list(configuration["member_names"]),
        stage_names=list(configuration["stage_names"]),
    )
    expected = list(configuration["feature_columns"])
    if list(features.columns) != expected:
        raise ValueError("v65 prediction-gap feature order differs from training")
    mean = np.asarray(configuration["feature_mean"], dtype=np.float64)
    scale = np.asarray(configuration["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(configuration["coefficients"], dtype=np.float64)
    if len(expected) != len(mean) or len(mean) != len(scale) or len(scale) != len(coefficients):
        raise ValueError("v65 prediction-gap parameter lengths differ")
    standardized = (features.to_numpy(np.float64) - mean) / scale
    raw = standardized @ coefficients
    regular = rows["game_type"].astype(str).eq(
        configuration.get("game_type_regular", "R"),
    ).to_numpy()
    regime_scale = np.where(
        regular,
        float(configuration["r_scale"]),
        float(configuration["f_scale"]),
    )
    correction = regime_scale * raw
    if not np.isfinite(correction).all():
        raise ValueError("v65 prediction-gap correction is non-finite")
    return correction
