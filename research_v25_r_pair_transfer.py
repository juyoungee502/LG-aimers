"""Screen regular-season numeric state x pitch-context residual tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_v25_f_pair_transfer import (
    BIN_COUNTS, CONTEXT_COLUMNS, SCALES, context_frame, pair_codes,
)
from research_v24_exhaustive_transfer import gain, table_direction


ROOT = Path(__file__).resolve().parent
SHRINKS = (400., 1600., 6400., 25600.)
MAX_NUMERIC = 40


def detail(y, base, direction, rows):
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
        if active.sum() >= 100:
            masks[f"month_{int(month)}"] = active
    return {
        name: gain(y[active], base[active], direction[active])
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
    contexts = context_frame(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    frame = features.iloc[positions].reset_index(drop=True)
    context = contexts.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    regular = rows["game_type"].eq("R").to_numpy()
    indices = {value: np.flatnonzero(regular & (year == value)) for value in (2023, 2024)}
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

    screened = json.loads(
        (ROOT / "research/v24_exhaustive_transfer.json").read_text(encoding="utf-8")
    )["top"]
    numeric_columns = []
    for item in screened:
        if item["kind"] == "numeric" and item["column"] not in numeric_columns:
            numeric_columns.append(item["column"])
        if len(numeric_columns) >= MAX_NUMERIC:
            break
    specs = [
        (column, context_column, bins, shrink)
        for column in numeric_columns for context_column in CONTEXT_COLUMNS
        for bins in BIN_COUNTS for shrink in SHRINKS
    ]
    reports = []
    for index, (column, context_column, bins, shrink) in enumerate(specs):
        directions = {}
        for label, source, valid in transfers:
            encoded = pair_codes(
                frame[column].iloc[source], frame[column].iloc[valid],
                context[context_column].iloc[source], context[context_column].iloc[valid], bins,
            )
            if encoded is None:
                directions = None
                break
            directions[label] = table_direction(
                encoded[0], encoded[1], y[source] - base[source], shrink,
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
                    "column": column, "context": context_column, "bins": bins,
                    "shrink": shrink, "scale": scale, "gains": gains,
                    "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
        if (index + 1) % 600 == 0:
            print(f"screened {index + 1}/{len(specs)}", flush=True)
    reports.sort(
        key=lambda item: (item["min_transfer"], item["mean_transfer"]), reverse=True,
    )

    strongest = {}
    for report in reports:
        strongest.setdefault((report["column"], report["context"]), report)
    source, valid = indices[2023], indices[2024]
    audits = []
    for report in strongest.values():
        encoded = pair_codes(
            frame[report["column"]].iloc[source], frame[report["column"]].iloc[valid],
            context[report["context"]].iloc[source], context[report["context"]].iloc[valid],
            int(report["bins"]),
        )
        direction = table_direction(
            encoded[0], encoded[1], y[source] - base[source], float(report["shrink"]),
        ) * float(report["scale"])
        values = detail(y[valid], base[valid], direction, rows.iloc[valid].reset_index(drop=True))
        audits.append({**report, "detail_2024": values, "min_2024_segment": min(values.values())})
    audits.sort(
        key=lambda item: (item["min_2024_segment"], item["min_transfer"], item["detail_2024"]["all"]),
        reverse=True,
    )
    result = {
        "numeric_columns": numeric_columns, "tested": len(specs) * len(SCALES),
        "positive_count": len(reports), "top": reports[:500], "audits": audits,
    }
    output = ROOT / "research/v25_r_pair_transfer.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "tested": result["tested"], "positive_count": len(reports),
        "top": reports[:40], "audits": audits[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
