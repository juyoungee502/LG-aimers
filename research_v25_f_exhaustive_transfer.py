"""Exhaustive post-break F-regime residual screen over v24.

Only 2023/2024 F rows are used, after the documented 2022->2023 regime break.
Every effect is fitted on an earlier labelled block and applied to later rows.
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
from research_v24_exhaustive_transfer import (
    encode_categorical, encode_numeric, gain, table_direction,
)


ROOT = Path(__file__).resolve().parent
BIN_COUNTS = (4, 8, 16, 32)
SHRINKS = (25., 100., 400., 1600., 6400.)
SCALES = (.25, .50, 1.00, 1.50)


def detail(y, base, candidate, rows):
    position = np.arange(len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        if active.sum() >= 40:
            masks[f"month_{int(month)}"] = active
    return {
        name: gain(y[active], base[active], candidate[active] - base[active])
        for name, active in masks.items()
    }


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target_series.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
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
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    frame = features.iloc[positions].reset_index(drop=True)
    category_frame = categorical.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    futures = rows["game_type"].eq("F").to_numpy()
    indices = {value: np.flatnonzero(futures & (year == value)) for value in (2023, 2024)}
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
    )

    specs = [
        ("numeric", column, bins, shrink)
        for column in frame.columns if column not in {"season", "game_month"}
        for bins in BIN_COUNTS for shrink in SHRINKS
    ]
    specs.extend(
        ("categorical", column, None, shrink)
        for column in category_frame.columns for shrink in SHRINKS
    )
    reports = []
    for candidate_index, (kind, column, bins, shrink) in enumerate(specs):
        values = frame[column] if kind == "numeric" else category_frame[column]
        directions = {}
        for label, source, valid in transfers:
            encoded = (
                encode_numeric(values.iloc[source], values.iloc[valid], int(bins))
                if kind == "numeric" else encode_categorical(values.iloc[source], values.iloc[valid])
            )
            if encoded is None:
                directions = None
                break
            directions[label] = table_direction(
                encoded[0], encoded[1], y[source] - base[source], float(shrink),
            )
        if directions is None:
            continue
        for scale in SCALES:
            gains = {
                label: gain(y[valid], base[valid], scale * directions[label])
                for label, _source, valid in transfers
            }
            if min(gains.values()) > 0:
                reports.append({
                    "kind": kind, "column": column, "bins": bins,
                    "shrink": shrink, "scale": scale, "gains": gains,
                    "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
        if (candidate_index + 1) % 600 == 0:
            print(f"screened {candidate_index + 1}/{len(specs)}", flush=True)

    reports.sort(key=lambda item: (item["min_transfer"], item["mean_transfer"]), reverse=True)
    strongest = {}
    for report in reports:
        strongest.setdefault((report["kind"], report["column"]), report)
    source, valid = indices[2023], indices[2024]
    audits = []
    for report in strongest.values():
        values = frame[report["column"]] if report["kind"] == "numeric" \
            else category_frame[report["column"]]
        encoded = (
            encode_numeric(values.iloc[source], values.iloc[valid], int(report["bins"]))
            if report["kind"] == "numeric" else encode_categorical(values.iloc[source], values.iloc[valid])
        )
        direction = table_direction(
            encoded[0], encoded[1], y[source] - base[source], float(report["shrink"]),
        ) * float(report["scale"])
        candidate = np.clip(base[valid] + direction, .005, .995)
        values_detail = detail(
            y[valid], base[valid], candidate, rows.iloc[valid].reset_index(drop=True),
        )
        audits.append({
            **report, "detail_2024": values_detail,
            "min_2024_segment": min(values_detail.values()),
        })
    audits.sort(
        key=lambda item: (item["min_2024_segment"], item["min_transfer"], item["detail_2024"]["all"]),
        reverse=True,
    )
    result = {
        "tested": len(specs) * len(SCALES), "positive_count": len(reports),
        "top": reports[:500], "audits": audits,
    }
    output = ROOT / "research/v25_f_exhaustive_transfer.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "tested": len(specs) * len(SCALES), "positive_count": len(reports),
        "top": reports[:50], "audits": audits[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
