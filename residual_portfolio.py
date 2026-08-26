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
    if name.startswith("derived_"):
        def col(column):
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)

        p1 = col("asof_pitcher_prev1_game_success_rate")
        p3 = col("asof_pitcher_prev3_game_success_rate")
        p5 = col("asof_pitcher_prev5_game_success_rate")
        m1 = col("asof_pitcher_prev1_game_middle_rate")
        m3 = col("asof_pitcher_prev3_game_middle_rate")
        m5 = col("asof_pitcher_prev5_game_middle_rate")
        if name == "derived_recent_success_mean":
            return .2 * p1 + .3 * p3 + .5 * p5
        if name == "derived_success_trend_1_3":
            return p1 - p3
        if name == "derived_recent_middle_mean":
            return .2 * m1 + .3 * m3 + .5 * m5
        if name == "derived_middle_range":
            values = np.column_stack([m1, m3, m5])
            finite = np.isfinite(values)
            safe_min = np.where(finite, values, np.inf).min(axis=1)
            safe_max = np.where(finite, values, -np.inf).max(axis=1)
            result = safe_max - safe_min
            result[~finite.any(axis=1)] = np.nan
            return result
        raise ValueError(f"Unknown derived residual feature: {name}")
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


def _categorical_column(frame, name):
    if name == "count_state":
        return (
            pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(np.int8) * 3
            + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(np.int8)
        )
    if name == "runner_count_code":
        return pd.to_numeric(
            frame["num_runners_on"], errors="coerce"
        ).fillna(0).astype(np.int8)
    return frame[name]


def _categorical_key(frame, keys):
    result = _categorical_column(frame, keys[0]).astype(str)
    for name in keys[1:]:
        result = result + "\x1f" + _categorical_column(frame, name).astype(str)
    return result


def freeze_generic_categorical_effect(frame, residual, keys, shrink, weight):
    key = _categorical_key(frame, keys)
    work = pd.DataFrame({"key": key, "residual": np.asarray(residual, np.float64)})
    table = work.groupby("key", sort=False, observed=True)["residual"].agg(["sum", "size"])
    table["value"] = table["sum"] / (table["size"] + float(shrink))
    mapped = key.map(table["value"]).fillna(0.).to_numpy(np.float64)
    table["value"] -= float(mapped.mean())
    return {
        "kind": "categorical", "features": list(keys),
        "keys": table.index.astype(str).tolist(),
        "table": table["value"].astype(float).tolist(), "weight": float(weight),
    }


def freeze_pair_effect(frame, residual, features, context, n_bins, shrink, weight):
    first, second = (_numeric(frame, name) for name in features)
    valid_first, valid_second = first[np.isfinite(first)], second[np.isfinite(second)]
    quantiles = np.linspace(0., 1., int(n_bins) + 1)[1:-1]
    edges_first = np.unique(np.quantile(valid_first, quantiles)) \
        if len(valid_first) else np.array([], dtype=np.float64)
    edges_second = np.unique(np.quantile(valid_second, quantiles)) \
        if len(valid_second) else np.array([], dtype=np.float64)

    def encode(value, edges):
        output = np.zeros(len(value), dtype=np.int32)
        finite = np.isfinite(value)
        output[finite] = np.searchsorted(edges, value[finite], side="right") + 1
        return output

    first_bin = encode(first, edges_first)
    second_bin = encode(second, edges_second)
    context_code, width = _context(frame, context)
    card_second = len(edges_second) + 2
    code = (first_bin * card_second + second_bin) * width + context_code
    size = (len(edges_first) + 2) * card_second * width
    sums = np.bincount(code, weights=np.asarray(residual, np.float64), minlength=size)
    counts = np.bincount(code, minlength=size)
    table = sums / (counts + float(shrink))
    table -= float(table[code].mean())
    return {
        "kind": "pair", "features": list(features), "context": context,
        "edges_first": edges_first.tolist(), "edges_second": edges_second.tolist(),
        "table": table.tolist(), "width": width, "weight": float(weight),
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
        elif effect["kind"] == "categorical":
            lookup = dict(zip(effect["keys"], effect["table"]))
            key = _categorical_key(frame, effect["features"])
            correction = key.map(lookup).fillna(0.).to_numpy(np.float64)
        elif effect["kind"] == "pair":
            first, second = (_numeric(frame, name) for name in effect["features"])
            edges_first = np.asarray(effect["edges_first"], dtype=np.float64)
            edges_second = np.asarray(effect["edges_second"], dtype=np.float64)

            def encode(value, edges):
                output = np.zeros(len(value), dtype=np.int32)
                finite = np.isfinite(value)
                output[finite] = np.searchsorted(edges, value[finite], side="right") + 1
                return output

            context_code, width = _context(frame, effect["context"])
            if width != int(effect["width"]):
                raise ValueError("Residual pair context width differs from training")
            code = (
                encode(first, edges_first) * (len(edges_second) + 2)
                + encode(second, edges_second)
            ) * width + context_code
            table = np.asarray(effect["table"], dtype=np.float64)
            correction = table[np.minimum(code, len(table) - 1)]
        else:
            raise ValueError(f"Unknown residual effect: {effect['kind']}")
        total += float(effect["weight"]) * correction
    active = frame["game_type"].astype(str).eq(
        configuration.get("game_type", "R")
    ).to_numpy()
    total[~active] = 0.
    return total
