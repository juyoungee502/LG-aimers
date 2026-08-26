"""Frozen, row-independent residual corrections used by v20 inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


LOG_FEATURES = {
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "li",
}
BASE_MAP = {
    "___": 0, "1__": 1, "_2_": 2, "__3": 3,
    "12_": 4, "1_3": 5, "_23": 6, "123": 7,
}


def _numeric(frame, name):
    value = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    if name in LOG_FEATURES:
        value = np.log1p(np.maximum(value, 0.))
    return value


def _context(frame, name):
    count = (
        pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.int32) * 3
        + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.int32)
    )
    if name == "none":
        return np.zeros(len(frame), dtype=np.int32), 1
    if name == "count":
        return count, 12
    phand = pd.to_numeric(frame["pitcher_hand"], errors="coerce").eq(1).to_numpy(np.int32)
    bhand = pd.to_numeric(frame["batter_hand"], errors="coerce").eq(1).to_numpy(np.int32)
    if name == "hands":
        return phand * 2 + bhand, 4
    if name == "count_hands":
        return count * 4 + phand * 2 + bhand, 48
    if name == "baseout":
        base = frame["base_state"].map(BASE_MAP).fillna(-1).to_numpy(np.int32)
        outs = pd.to_numeric(frame["outs_before"], errors="coerce").fillna(0).to_numpy(np.int32)
        return np.maximum(base * 3 + outs, 0), 24
    raise ValueError(f"Unknown context: {name}")


def freeze_numeric_effect(frame, residual, feature, context, n_bins, shrink, weight):
    value = _numeric(frame, feature)
    valid = value[np.isfinite(value)]
    edges = np.unique(
        np.quantile(valid, np.linspace(0., 1., int(n_bins) + 1)[1:-1])
    ) if len(valid) else np.array([], dtype=np.float64)
    bins = np.zeros(len(frame), dtype=np.int32)
    finite = np.isfinite(value)
    bins[finite] = np.searchsorted(edges, value[finite], side="right") + 1
    ctx, width = _context(frame, context)
    code = bins * width + ctx
    size = int((len(edges) + 2) * width)
    sums = np.bincount(code, weights=np.asarray(residual, np.float64), minlength=size)
    counts = np.bincount(code, minlength=size)
    table = sums / (counts + float(shrink))
    table -= float(table[code].mean())
    return {
        "kind": "numeric", "feature": feature, "context": context,
        "edges": edges.tolist(), "table": table.tolist(), "width": width,
        "weight": float(weight),
    }


def freeze_categorical_effect(frame, residual, shrink, weight):
    key = frame["batter_id"].astype(str) + ":" + frame["pitcher_hand"].astype(str)
    work = pd.DataFrame({"key": key, "residual": np.asarray(residual, np.float64)})
    table = work.groupby("key", sort=False, observed=True)["residual"].agg(["sum", "size"])
    table["value"] = table["sum"] / (table["size"] + float(shrink))
    mapped = key.map(table["value"]).fillna(0.).to_numpy(np.float64)
    table["value"] -= float(mapped.mean())
    return {
        "kind": "categorical_batter_phand", "keys": table.index.astype(str).tolist(),
        "table": table["value"].astype(float).tolist(), "weight": float(weight),
    }


def apply_frozen_portfolio(frame, configuration):
    total = np.zeros(len(frame), dtype=np.float64)
    for effect in configuration["effects"]:
        if effect["kind"] == "numeric":
            value = _numeric(frame, effect["feature"])
            edges = np.asarray(effect["edges"], dtype=np.float64)
            bins = np.zeros(len(frame), dtype=np.int32)
            finite = np.isfinite(value)
            bins[finite] = np.searchsorted(edges, value[finite], side="right") + 1
            ctx, width = _context(frame, effect["context"])
            if width != int(effect["width"]):
                raise ValueError("Residual context width differs from training")
            code = bins * width + ctx
            table = np.asarray(effect["table"], dtype=np.float64)
            correction = table[np.minimum(code, len(table) - 1)]
        elif effect["kind"] == "categorical_batter_phand":
            lookup = dict(zip(effect["keys"], effect["table"]))
            key = frame["batter_id"].astype(str) + ":" + frame["pitcher_hand"].astype(str)
            correction = key.map(lookup).fillna(0.).to_numpy(np.float64)
        else:
            raise ValueError(f"Unknown residual effect: {effect['kind']}")
        total += float(effect["weight"]) * correction
    active = frame["game_type"].astype(str).eq(
        configuration.get("game_type", "R")
    ).to_numpy()
    total[~active] = 0.
    return total
