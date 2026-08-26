"""Frozen row-wise probability-shape corrections used by v23 inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gate(frame, name):
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    count = (
        pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.int32) * 3
        + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.int32)
    )
    if name.startswith("regular_count_"):
        return regular & (count == int(name.rsplit("_", 1)[1]))
    if name.startswith("regular_runners_"):
        runners = pd.to_numeric(
            frame["num_runners_on"], errors="coerce"
        ).fillna(0).to_numpy(np.int32)
        return regular & (runners == int(name.rsplit("_", 1)[1]))
    raise ValueError(f"Unknown probability residual gate: {name}")


def _shape(prediction, name):
    p = np.clip(np.asarray(prediction, dtype=np.float64), .005, .995)
    if name == "constant":
        return np.ones(len(p), dtype=np.float64)
    if name == "uncertainty":
        return p * (1. - p)
    if name == "quadratic":
        return (p - .5) ** 2
    raise ValueError(f"Unknown probability residual shape: {name}")


def apply_probability_portfolio(frame, prediction, configuration):
    prediction = np.asarray(prediction, dtype=np.float64)
    correction = np.zeros(len(prediction), dtype=np.float64)
    for effect in configuration["effects"]:
        value = _shape(prediction, effect["shape"]) - float(effect["center"])
        correction += float(effect["weight"]) * value * _gate(frame, effect["gate"])
    return correction
