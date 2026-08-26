"""Frozen row-wise component reblends used by v22 inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def history_expert(features, prior):
    specs = (
        ("asof_pitcher_prev1_game_success_rate", .12),
        ("asof_pitcher_prev3_game_success_rate", .28),
        ("asof_pitcher_prev5_game_success_rate", .20),
        ("pitcher_season_success_s100", .22),
        ("asof_pitcher_success_rate", .10),
        ("asof_batter_success_rate", .08),
    )
    total = np.zeros(len(features), dtype=np.float64)
    weight = np.zeros(len(features), dtype=np.float64)
    for column, component_weight in specs:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        total[finite] += component_weight * values[finite]
        weight[finite] += component_weight
    return np.divide(
        total, weight, out=np.full(len(features), float(prior)), where=weight > 0,
    )


def _gate(frame, name):
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    count = (
        pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.int32) * 3
        + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.int32)
    )
    if name == "other":
        return count % 3 != 2
    if name == "two_strike":
        return count % 3 == 2
    if name.startswith("regular_count_"):
        return regular & (count == int(name.rsplit("_", 1)[1]))
    if name.startswith("regular_runners_"):
        runners = pd.to_numeric(
            frame["num_runners_on"], errors="coerce"
        ).fillna(0).to_numpy(np.int32)
        return regular & (runners == int(name.rsplit("_", 1)[1]))
    raise ValueError(f"Unknown component residual gate: {name}")


def apply_component_portfolio(frame, components, anchor, configuration):
    anchor = np.asarray(anchor, dtype=np.float64)
    correction = np.zeros(len(frame), dtype=np.float64)
    for effect in configuration["effects"]:
        component = np.asarray(components[effect["prediction"]], dtype=np.float64)
        if len(component) != len(anchor):
            raise ValueError("Component residual prediction length differs")
        direction = np.nan_to_num(component - anchor, nan=0.)
        correction += float(effect["weight"]) * direction * _gate(frame, effect["gate"])
    return correction
