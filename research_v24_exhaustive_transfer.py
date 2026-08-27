"""Exhaustively screen one-dimensional residual effects over four transfers.

Every table is fitted on an earlier block and frozen before it is applied to a
later block.  This is deliberately a rejection screen: a candidate must improve
all four transfers before it can be considered for a production portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)


ROOT = Path(__file__).resolve().parent
BIN_COUNTS = (4, 8, 16)
SHRINKS = (100., 400., 1600., 6400.)
SCALES = (.25, .5, 1.)


def gain(target, base, correction):
    target = np.asarray(target, dtype=float)
    base = np.asarray(base, dtype=float)
    candidate = np.clip(base + np.asarray(correction, dtype=float), .005, .995)
    reference = float(target.mean() * (1. - target.mean()))
    return float(100000. * (
        np.mean((target - base) ** 2) - np.mean((target - candidate) ** 2)
    ) / reference)


def encode_numeric(source, query, bins):
    source = pd.to_numeric(source, errors="coerce").to_numpy(float)
    query = pd.to_numeric(query, errors="coerce").to_numpy(float)
    finite = np.isfinite(source)
    if finite.sum() < 100 or np.nanmin(source[finite]) == np.nanmax(source[finite]):
        return None
    edges = np.unique(np.quantile(
        source[finite], np.linspace(0., 1., bins + 1)[1:-1],
    ))
    if not len(edges):
        return None
    source_code = np.zeros(len(source), dtype=np.int32)
    query_code = np.zeros(len(query), dtype=np.int32)
    source_code[finite] = np.searchsorted(edges, source[finite], side="right") + 1
    query_finite = np.isfinite(query)
    query_code[query_finite] = np.searchsorted(
        edges, query[query_finite], side="right",
    ) + 1
    return source_code, query_code


def encode_categorical(source, query):
    source_text = source.astype("string").fillna("<NA>")
    query_text = query.astype("string").fillna("<NA>")
    unique = pd.Index(source_text.unique())
    if len(unique) < 2 or len(unique) > 5000:
        return None
    source_code = unique.get_indexer(source_text).astype(np.int32) + 1
    query_code = unique.get_indexer(query_text).astype(np.int32) + 1
    query_code[query_code < 1] = 0
    return source_code, query_code


def table_direction(source_code, query_code, residual, shrink):
    size = int(max(source_code.max(initial=0), query_code.max(initial=0)) + 1)
    sums = np.bincount(source_code, weights=residual, minlength=size)
    counts = np.bincount(source_code, minlength=size)
    table = sums / (counts + float(shrink))
    source_value = table[source_code]
    table -= float(source_value.mean())
    return table[query_code]


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    # Raw string codes are useful for frozen categorical residual tables, while
    # engineered numeric columns expose the current-season algebra.
    categorical = raw[[
        column for column in raw.columns
        if raw[column].dtype == "object" or column.endswith("_id")
    ]].copy()
    for column in (
        "balls_before", "strikes_before", "outs_before", "inning",
        "pitcher_hand", "batter_hand", "num_runners_on",
    ):
        categorical[column] = raw[column]

    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    if not np.allclose(target[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    frame = features.iloc[positions].reset_index(drop=True)
    category_frame = categorical.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    regular = rows["game_type"].eq("R").to_numpy()

    regular_indices = {value: np.flatnonzero(regular & (year == value)) for value in (2023, 2024)}
    halves = {
        (value, half): index[:len(index) // 2] if half == 1 else index[len(index) // 2:]
        for value, index in regular_indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", regular_indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", regular_indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
    )

    reports = []
    candidate_specs = [
        ("numeric", column, bins, shrink)
        for column in frame.columns
        if column not in {"season", "game_month"}
        for bins in BIN_COUNTS for shrink in SHRINKS
    ]
    candidate_specs.extend(
        ("categorical", column, None, shrink)
        for column in category_frame.columns for shrink in SHRINKS
    )
    print(
        f"screening candidates={len(candidate_specs)} numeric_features={len(frame.columns)} "
        f"categorical_features={len(category_frame.columns)}", flush=True,
    )
    for candidate_index, (kind, column, bins, shrink) in enumerate(candidate_specs):
        values = frame[column] if kind == "numeric" else category_frame[column]
        directions = {}
        failed = False
        for label, source, valid in transfers:
            encoded = (
                encode_numeric(values.iloc[source], values.iloc[valid], int(bins))
                if kind == "numeric"
                else encode_categorical(values.iloc[source], values.iloc[valid])
            )
            if encoded is None:
                failed = True
                break
            source_code, valid_code = encoded
            residual = y[source] - base[source]
            directions[label] = table_direction(
                source_code, valid_code, residual, float(shrink),
            )
        if failed:
            continue
        for scale in SCALES:
            gains = {
                label: gain(y[valid], base[valid], scale * directions[label])
                for label, _source, valid in transfers
            }
            reports.append({
                "kind": kind, "column": column, "bins": bins,
                "shrink": shrink, "scale": scale, "gains": gains,
                "min_transfer": min(gains.values()),
                "mean_transfer": float(np.mean(list(gains.values()))),
            })
        if (candidate_index + 1) % 500 == 0:
            print(f"screened {candidate_index + 1}/{len(candidate_specs)}", flush=True)

    reports.sort(
        key=lambda value: (value["min_transfer"], value["mean_transfer"]),
        reverse=True,
    )
    positive = [value for value in reports if value["min_transfer"] > 0.]
    output = ROOT / "research/v24_exhaustive_transfer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "positive_count": len(positive), "top": reports[:300],
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "tested": len(reports), "positive_count": len(positive),
        "top": reports[:60],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
