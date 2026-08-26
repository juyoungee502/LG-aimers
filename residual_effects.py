"""Row-independent empirical-Bayes residual effects for v12 inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


EFFECT_CONFIG = (
    {"name": "pitcher_main", "kind": "main", "column": "pitcher_id", "k": 2000., "weight": .18},
    {"name": "batter_main", "kind": "main", "column": "batter_id", "k": 10000., "weight": 1.00},
    {"name": "pitcher_same_hand", "kind": "diff", "context": "same_hand", "k": 500., "weight": 1.00},
    {"name": "pitcher_two_strike", "kind": "diff", "context": "two_strike", "k": 1000., "weight": 1.20},
    {"name": "pitcher_runners", "kind": "diff", "context": "runners", "k": 500., "weight": .70},
    {"name": "pitcher_ball_adv", "kind": "diff", "context": "ball_adv", "k": 500., "weight": .85},
    {"name": "count_level", "kind": "count", "k": 0., "weight": .30},
)


def _context(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "same_hand":
        return df["pitcher_hand"].eq(df["batter_hand"]).to_numpy(np.int8)
    if name == "two_strike":
        return df["strikes_before"].eq(2).to_numpy(np.int8)
    if name == "runners":
        return df["num_runners_on"].gt(0).to_numpy(np.int8)
    if name == "ball_adv":
        return df["balls_before"].gt(df["strikes_before"]).to_numpy(np.int8)
    raise ValueError(f"Unknown residual context: {name}")


def _center_table(keys: np.ndarray, table: pd.Series) -> pd.Series:
    mapped = pd.Series(keys).map(table).fillna(0.).to_numpy(np.float64)
    return table - float(mapped.mean())


def build_residual_effects(df: pd.DataFrame, residual: np.ndarray) -> list[dict]:
    """Build fixed lookup tables from strictly out-of-fold residuals."""
    residual = np.asarray(residual, dtype=np.float64)
    if len(df) != len(residual):
        raise ValueError("Residual rows differ from source rows")
    effects = []
    for config in EFFECT_CONFIG:
        spec = dict(config)
        kind = spec["kind"]
        if kind == "main":
            keys = df[spec["column"]].to_numpy()
            grouped = pd.DataFrame({"key": keys, "residual": residual}).groupby("key")["residual"].agg(["sum", "size"])
            table = _center_table(keys, grouped["sum"] / (grouped["size"] + spec["k"]))
            spec["table"] = {str(int(key)): float(value) for key, value in table.items()}
        elif kind == "diff":
            pitcher = df["pitcher_id"].to_numpy()
            context = _context(df, spec["context"])
            grouped = pd.DataFrame({"pitcher": pitcher, "context": context, "residual": residual}).groupby(
                ["pitcher", "context"]
            )["residual"].agg(["mean", "size"]).unstack()
            for statistic in ("mean", "size"):
                for value in (0, 1):
                    if (statistic, value) not in grouped:
                        grouped[(statistic, value)] = 0.
            n0 = grouped[("size", 0)].fillna(0.)
            n1 = grouped[("size", 1)].fillna(0.)
            effective_n = n0 * n1 / (n0 + n1).replace(0., np.nan)
            delta = (grouped[("mean", 1)] - grouped[("mean", 0)]) * effective_n / (effective_n + spec["k"])
            prevalence = n1 / (n0 + n1).replace(0., np.nan)
            values = pd.DataFrame({"v0": -prevalence * delta, "v1": (1. - prevalence) * delta}).dropna()
            v0 = pd.Series(pitcher).map(values["v0"]).fillna(0.).to_numpy()
            v1 = pd.Series(pitcher).map(values["v1"]).fillna(0.).to_numpy()
            center = float(np.where(context == 1, v1, v0).mean())
            spec["table"] = {
                str(int(key)): [float(row.v0 - center), float(row.v1 - center)]
                for key, row in values.iterrows()
            }
        elif kind == "count":
            keys = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
            grouped = pd.DataFrame({"key": keys, "residual": residual}).groupby("key")["residual"].mean()
            table = _center_table(keys, grouped)
            spec["table"] = {str(int(key)): float(value) for key, value in table.items()}
        else:
            raise ValueError(f"Unknown residual effect kind: {kind}")
        effects.append(spec)
    return effects


def apply_residual_effects(df: pd.DataFrame, effects: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply frozen tables using only the current row's values."""
    total = np.zeros(len(df), dtype=np.float64)
    components = {}
    for spec in effects:
        kind = spec["kind"]
        table = spec["table"]
        if kind == "main":
            values = df[spec["column"]].astype(str).map(table).fillna(0.).to_numpy(np.float64)
        elif kind == "diff":
            context = _context(df, spec["context"])
            pairs = df["pitcher_id"].astype(str).map(table)
            v0 = pairs.map(lambda pair: pair[0] if isinstance(pair, list) else 0.).to_numpy(np.float64)
            v1 = pairs.map(lambda pair: pair[1] if isinstance(pair, list) else 0.).to_numpy(np.float64)
            values = np.where(context == 1, v1, v0)
        elif kind == "count":
            keys = (df["balls_before"] * 4 + df["strikes_before"]).astype(str)
            values = keys.map(table).fillna(0.).to_numpy(np.float64)
        else:
            raise ValueError(f"Unknown residual effect kind: {kind}")
        components[spec["name"]] = values
        total += float(spec["weight"]) * values
    return total, components
