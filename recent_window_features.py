"""Recover confidence information hidden in recent-game rate fractions.

The official inputs expose success and middle rates over the previous 1, 3,
and 5 games, but not their pitch counts.  Both rates in a window are rounded
fractions with the same denominator.  Their least common reduced denominator
therefore recovers the true pitch count whenever the three integer counts are
coprime, and otherwise supplies a conservative lower bound.  Every feature in
this module is computed from one row only.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm

import numpy as np
import pandas as pd


WINDOW_LIMITS = {1: 180, 3: 450, 5: 750}


def _pair_denominators(
    success: pd.Series, middle: pd.Series, max_denominator: int,
) -> np.ndarray:
    """Return the common reduced denominator for each rate pair."""
    pairs = pd.DataFrame({
        "success": pd.to_numeric(success, errors="coerce"),
        "middle": pd.to_numeric(middle, errors="coerce"),
    })
    valid = pairs.notna().all(axis=1).to_numpy()
    output = np.zeros(len(pairs), dtype=np.float32)
    if not valid.any():
        return output

    matrix = pairs.loc[valid, ["success", "middle"]].to_numpy(np.float64)
    unique, inverse = np.unique(matrix, axis=0, return_inverse=True)
    denominators = np.zeros(len(unique), dtype=np.float32)
    for index, (success_rate, middle_rate) in enumerate(unique):
        success_denominator = Fraction(float(success_rate)).limit_denominator(
            max_denominator
        ).denominator
        middle_denominator = Fraction(float(middle_rate)).limit_denominator(
            max_denominator
        ).denominator
        common = lcm(success_denominator, middle_denominator)
        if common <= max_denominator:
            denominators[index] = float(common)
    output[valid] = denominators[inverse]
    return output


def _count(rate: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(rate, nan=0., posinf=0., neginf=0.)
    return np.rint(values * denominator).astype(np.float32)


def recent_window_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Build row-independent recent-window denominator and contrast features."""
    result: dict[str, np.ndarray] = {}
    denominators: dict[int, np.ndarray] = {}
    success_counts: dict[int, np.ndarray] = {}
    middle_counts: dict[int, np.ndarray] = {}
    success_rates: dict[int, np.ndarray] = {}
    middle_rates: dict[int, np.ndarray] = {}
    career = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(.5).to_numpy(np.float64)

    for window, limit in WINDOW_LIMITS.items():
        success_column = f"asof_pitcher_prev{window}_game_success_rate"
        middle_column = f"asof_pitcher_prev{window}_game_middle_rate"
        denominator = _pair_denominators(
            rows[success_column], rows[middle_column], limit,
        )
        success = pd.to_numeric(
            rows[success_column], errors="coerce"
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            rows[middle_column], errors="coerce"
        ).to_numpy(np.float64)
        success_count = _count(success, denominator)
        middle_count = _count(middle, denominator)
        observed = denominator > 0.

        denominators[window] = denominator
        success_counts[window] = success_count
        middle_counts[window] = middle_count
        success_rates[window] = success
        middle_rates[window] = middle
        result[f"recent{window}_reduced_n"] = denominator
        result[f"recent{window}_log_reduced_n"] = np.log1p(denominator)
        result[f"recent{window}_success_count"] = success_count
        result[f"recent{window}_middle_count"] = middle_count
        result[f"recent{window}_fraction_observed"] = observed.astype(np.float32)
        for strength in (10., 25., 50., 100.):
            smoothed = np.divide(
                success_count + strength * career,
                denominator + strength,
                out=career.copy(),
                where=observed,
            )
            result[f"recent{window}_success_s{int(strength)}"] = smoothed
            result[f"recent{window}_weight_s{int(strength)}"] = (
                denominator / (denominator + strength)
            )

    monotone = (
        (denominators[1] > 0.)
        & (denominators[1] <= denominators[3])
        & (denominators[3] <= denominators[5])
    )
    result["recent_fraction_n_monotone"] = monotone.astype(np.float32)
    result["recent_fraction_n_ratio_1_3"] = np.divide(
        denominators[1], denominators[3],
        out=np.zeros(len(rows), np.float32), where=denominators[3] > 0.,
    )
    result["recent_fraction_n_ratio_3_5"] = np.divide(
        denominators[3], denominators[5],
        out=np.zeros(len(rows), np.float32), where=denominators[5] > 0.,
    )

    # Subtract nested windows to estimate games 2-3 and 4-5.  GCD reduction can
    # make a denominator only a lower bound, so invalid differences are marked
    # missing instead of being forced into a plausible range.
    for short, long, name in ((1, 3, "games2_3"), (3, 5, "games4_5")):
        n = denominators[long] - denominators[short]
        success = success_counts[long] - success_counts[short]
        middle = middle_counts[long] - middle_counts[short]
        valid = (
            (n > 0.) & (success >= 0.) & (success <= n)
            & (middle >= 0.) & (middle <= n)
        )
        result[f"recent_{name}_reduced_n"] = np.where(valid, n, np.nan)
        result[f"recent_{name}_success_rate"] = np.divide(
            success, n, out=np.full(len(rows), np.nan, np.float32), where=valid,
        )
        result[f"recent_{name}_middle_rate"] = np.divide(
            middle, n, out=np.full(len(rows), np.nan, np.float32), where=valid,
        )
        result[f"recent_{name}_valid"] = valid.astype(np.float32)

    # Reliability-weighted momentum: large reduced denominators allow the raw
    # recent difference to influence the model more strongly.
    for short, long in ((1, 3), (3, 5), (1, 5)):
        reliability = np.minimum(
            denominators[short] / (denominators[short] + 25.),
            denominators[long] / (denominators[long] + 25.),
        )
        result[f"recent_success_reliable_trend_{short}_{long}"] = (
            np.nan_to_num(success_rates[short] - success_rates[long]) * reliability
        )
        result[f"recent_middle_reliable_trend_{short}_{long}"] = (
            np.nan_to_num(middle_rates[short] - middle_rates[long]) * reliability
        )

    return pd.DataFrame(result, index=rows.index, dtype=np.float32)
