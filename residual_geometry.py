"""Deterministic geometry transforms for empirical-Bayes player residual tables."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def _row_vector(values, positions):
    output = np.zeros(len(positions), dtype=np.float64)
    hit = positions >= 0
    output[hit] = values[positions[hit]]
    return output


def _safe_scale(reference, direction):
    denominator = float(np.std(direction))
    return float(np.std(reference) / denominator) if denominator > 0 else 0.0


def reshape_main_effect(
    df: pd.DataFrame,
    residual: np.ndarray,
    effects: list[dict],
    *,
    effect_name: str,
    column: str,
    shape_k: float | None = None,
    shape_step: float = 0.0,
    exposure_step: float = 0.0,
    log_exposure_step: float = 0.0,
) -> list[dict]:
    """Move a player main-effect table along normalized, orthogonal count axes.

    The transformation uses training-row counts only. The frozen result remains
    a row-independent lookup table at inference time.
    """
    result = copy.deepcopy(effects)
    spec = next(item for item in result if item["name"] == effect_name)
    ids = np.asarray(sorted(int(key) for key in spec["table"]), dtype=np.int64)
    base = np.asarray([spec["table"][str(int(key))] for key in ids], dtype=np.float64)
    position_map = {int(key): index for index, key in enumerate(ids)}
    positions = np.asarray([position_map.get(int(key), -1) for key in df[column]], dtype=np.int64)

    grouped = pd.DataFrame({
        "key": df[column].to_numpy(),
        "residual": np.asarray(residual, dtype=np.float64),
    }).groupby("key")["residual"].agg(["sum", "size"]).reindex(ids)
    counts = grouped["size"].to_numpy(np.float64)
    shaped = base.copy()
    base_rows = _row_vector(base, positions)
    if shape_k is not None and shape_step:
        alternative = np.array(
            grouped["sum"] / (counts + shape_k), dtype=np.float64, copy=True
        )
        alternative -= float(_row_vector(alternative, positions).mean())
        alternative *= _safe_scale(base_rows, _row_vector(alternative, positions))
        shaped = base + float(shape_step) * (alternative - base)

    shaped_rows = _row_vector(shaped, positions)
    raw_n = counts - float(counts.mean())
    raw_n_rows = _row_vector(raw_n, positions)
    denominator = float(np.dot(shaped_rows, shaped_rows))
    n_direction = raw_n.copy()
    if denominator > 0:
        n_direction -= float(np.dot(raw_n_rows, shaped_rows) / denominator) * shaped
    n_rows = _row_vector(n_direction, positions)
    final = shaped + float(exposure_step) * _safe_scale(shaped_rows, n_rows) * n_direction

    if log_exposure_step:
        log_direction = np.log1p(counts) - float(np.log1p(counts).mean())
        # Covariance orthogonalization preserves the already measured shape and
        # linear-exposure coordinates while isolating the concave count signal.
        for basis in (shaped, n_direction):
            direction_rows = _row_vector(log_direction, positions)
            basis_rows = _row_vector(basis, positions)
            direction_rows -= direction_rows.mean()
            basis_rows -= basis_rows.mean()
            basis_norm = float(np.dot(basis_rows, basis_rows))
            if basis_norm > 0:
                log_direction -= float(np.dot(direction_rows, basis_rows) / basis_norm) * basis
        log_rows = _row_vector(log_direction, positions)
        final += (
            float(log_exposure_step)
            * _safe_scale(shaped_rows, log_rows)
            * log_direction
        )

    spec["table"] = {
        str(int(key)): float(value) for key, value in zip(ids, final)
    }
    spec["geometry"] = {
        "shape_k": shape_k,
        "shape_step": float(shape_step),
        "exposure_step": float(exposure_step),
        "log_exposure_step": float(log_exposure_step),
    }
    return result


def apply_v13_geometry(df, residual, effects):
    """Apply the robust v13 table coordinates selected by rolling transfer."""
    result = reshape_main_effect(
        df, residual, effects,
        effect_name="batter_main", column="batter_id",
        shape_k=500.0, shape_step=-3.0,
        exposure_step=0.40, log_exposure_step=0.16,
    )
    return reshape_main_effect(
        df, residual, result,
        effect_name="pitcher_main", column="pitcher_id",
        exposure_step=0.50,
    )
