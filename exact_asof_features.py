"""Row-local exact-count representations of the official as-of columns.

The published rates are cumulative fractions.  Multiplying them by their
official denominators recovers the corresponding integer counts without
looking at another evaluation row.  Tree models otherwise have to approximate
these rational relationships through many independent numeric splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PITCHER_RATE_COLUMNS = (
    "success", "reverse", "middle", "ball", "strike",
)
PITCHMIX_RATE_COLUMNS = ("fastball", "breaking", "offspeed")


def _count(n: pd.Series, rate: pd.Series) -> np.ndarray:
    n_values = n.fillna(0).to_numpy(np.float64, copy=False)
    rate_values = rate.fillna(0).to_numpy(np.float64, copy=False)
    return np.rint(n_values * rate_values)


def exact_asof_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Return features computed solely from values in the current row."""
    pitcher_n = rows["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    batter_n = rows["asof_batter_n"].fillna(0).to_numpy(np.float64)
    mix_n = rows["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
    values: dict[str, np.ndarray] = {}

    pitcher_counts = {}
    for name in PITCHER_RATE_COLUMNS:
        count = _count(
            rows["asof_pitcher_n"], rows[f"asof_pitcher_{name}_rate"],
        )
        pitcher_counts[name] = count
        values[f"exact_pitcher_{name}_count"] = count
        values[f"exact_pitcher_{name}_log_count"] = np.log1p(count)

    batter_success = _count(
        rows["asof_batter_n"], rows["asof_batter_success_rate"],
    )
    batter_middle = _count(
        rows["asof_batter_n"], rows["asof_batter_middle_rate"],
    )
    values["exact_batter_success_count"] = batter_success
    values["exact_batter_middle_count"] = batter_middle
    values["exact_batter_failure_count"] = np.maximum(
        0.0, batter_n - batter_success,
    )

    mix_counts = {}
    for name in PITCHMIX_RATE_COLUMNS:
        count = _count(
            rows["asof_pitcher_pitchmix_n"],
            rows[f"asof_pitcher_{name}_rate"],
        )
        mix_counts[name] = count
        values[f"exact_pitchmix_{name}_count"] = count
        values[f"exact_pitchmix_{name}_log_count"] = np.log1p(count)

    values["exact_pitcher_failure_count"] = np.maximum(
        0.0, pitcher_n - pitcher_counts["success"],
    )
    values["exact_pitcher_unclassified_location_count"] = np.maximum(
        0.0,
        pitcher_n
        - pitcher_counts["success"]
        - pitcher_counts["reverse"]
        - pitcher_counts["middle"],
    )
    values["exact_pitcher_uncalled_count"] = np.maximum(
        0.0, pitcher_n - pitcher_counts["ball"] - pitcher_counts["strike"],
    )
    values["exact_pitchmix_other_count"] = np.maximum(
        0.0,
        mix_n
        - mix_counts["fastball"]
        - mix_counts["breaking"]
        - mix_counts["offspeed"],
    )

    # Contrasts expose the latent failure composition directly.  They retain
    # the exact denominator information instead of repeating the raw rates.
    values["exact_success_minus_reverse_count"] = (
        pitcher_counts["success"] - pitcher_counts["reverse"]
    )
    values["exact_success_minus_middle_count"] = (
        pitcher_counts["success"] - pitcher_counts["middle"]
    )
    values["exact_strike_minus_ball_count"] = (
        pitcher_counts["strike"] - pitcher_counts["ball"]
    )
    values["exact_pitcher_minus_batter_success_count"] = (
        pitcher_counts["success"] - batter_success
    )

    return pd.DataFrame(values, index=rows.index, dtype=np.float32)
