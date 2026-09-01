"""Frozen row-local inference for the original v67 count geometry."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_count_geometry(rows: pd.DataFrame, configuration: dict) -> np.ndarray:
    """Look up a pitcher-by-count correction learned from official train rows."""
    pitcher_ids = pd.Index([str(value) for value in configuration["pitcher_ids"]])
    values = np.asarray(configuration["values"], dtype=np.float64)
    if values.shape != (len(pitcher_ids), 12):
        raise ValueError("v67 count-geometry table must have 12 cells per pitcher")

    positions = pitcher_ids.get_indexer(rows["pitcher_id"].astype(str))
    balls = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(0).to_numpy(int)
    strikes = pd.to_numeric(
        rows["strikes_before"], errors="coerce",
    ).fillna(0).to_numpy(int)
    counts = balls * 3 + strikes
    valid_count = (counts >= 0) & (counts < 12)
    known = (positions >= 0) & valid_count
    correction = np.zeros(len(rows), dtype=np.float64)
    correction[known] = values[positions[known], counts[known]]
    correction *= float(configuration["scale"])
    if not np.isfinite(correction).all():
        raise ValueError("v67 count-geometry correction is non-finite")
    return correction
