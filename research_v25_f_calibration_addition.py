"""Audit probability-table additions after the robust F portfolio."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_v25_f_combined_portfolio import candidate_name, direction
from research_v25_f_pair_transfer import context_frame
from research_v25_f_portfolio import exact_metrics, masks
from research_v25_probability_tables import codes
from research_v24_exhaustive_transfer import table_direction


ROOT = Path(__file__).resolve().parent
GRID = (0., .25, .50, .75, 1.)


def probability_direction(spec, base, context, source, valid, residual):
    encoded = codes(
        pd.Series(base), context, source, valid, spec["context"], int(spec["bins"]),
    )
    return table_direction(
        encoded[0], encoded[1], residual[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    numeric_all = engineer_features(raw, *bases, global_prior=float(target_series.mean()))
    add_training_component_features(numeric_all, raw)
    numeric_all = add_state_interactions(numeric_all)
    categorical_all = raw[[
        column for column in raw.columns
        if raw[column].dtype == "object" or column.endswith("_id")
    ]].copy()
    for column in (
        "balls_before", "strikes_before", "outs_before", "inning",
        "pitcher_hand", "batter_hand", "num_runners_on",
    ):
        categorical_all[column] = raw[column]
    context_all = context_frame(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    numeric = numeric_all.iloc[positions].reset_index(drop=True)
    categorical = categorical_all.iloc[positions].reset_index(drop=True)
    context = context_all.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    residual = y - base
    year = oof["season"].astype(int)
    future = rows["game_type"].eq("F").to_numpy()
    indices = {value: np.flatnonzero(future & (year == value)) for value in (2023, 2024)}
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

    portfolio = json.loads(
        (ROOT / "research/v25_f_combined_portfolio.json").read_text(encoding="utf-8")
    )
    baseline = max(
        (
            item for item in portfolio["solutions"]
            if item["cap"] == 3. and item["rounding"] == .05
        ),
        key=lambda item: (item["min_strict"], item["gain_2024"]),
    )
    candidates = portfolio["candidates"]
    weights = np.array([
        baseline["weights"].get(candidate_name(candidate), 0.)
        for candidate in candidates
    ])
    probability = json.loads(
        (ROOT / "research/v25_probability_tables.json").read_text(encoding="utf-8")
    )["F"]["top"]
    unique = {}
    for spec in probability:
        unique.setdefault(spec["context"], spec)
    calibration_specs = list(unique.values())

    blocks = {}
    for label, source, valid in transfers:
        base_directions = np.stack([
            direction(candidate, numeric, categorical, context, source, valid, residual)
            for candidate in candidates
        ])
        calibration = np.stack([
            probability_direction(spec, base, context, source, valid, residual)
            for spec in calibration_specs
        ])
        blocks[label] = {
            "y": y[valid], "base": base[valid],
            "baseline": weights @ base_directions, "calibration": calibration,
            "masks": masks(rows.iloc[valid].reset_index(drop=True), label),
        }
    source, valid = indices[2023], indices[2024]
    base_directions = np.stack([
        direction(candidate, numeric, categorical, context, source, valid, residual)
        for candidate in candidates
    ])
    calibration = np.stack([
        probability_direction(spec, base, context, source, valid, residual)
        for spec in calibration_specs
    ])
    blocks["2024"] = {
        "y": y[valid], "base": base[valid],
        "baseline": weights @ base_directions, "calibration": calibration,
        "masks": masks(rows.iloc[valid].reset_index(drop=True), "2024"),
    }

    results = []
    for values in itertools.product(GRID, repeat=len(calibration_specs)):
        metrics = {}
        for block in blocks.values():
            correction = block["baseline"] + np.tensordot(
                values, block["calibration"], axes=1,
            )
            metrics.update(exact_metrics(
                block["y"], block["base"], correction[None, :], np.ones(1),
                block["masks"],
            ))
        results.append({
            "calibration_weights": {
                spec["context"]: float(value)
                for spec, value in zip(calibration_specs, values) if value > 0
            },
            "min_strict": float(min(metrics.values())),
            "gain_2024": float(metrics["2024/all"]), "metrics": metrics,
        })
    robust = sorted(
        results, key=lambda item: (item["min_strict"], item["gain_2024"]), reverse=True,
    )
    positive = sorted(
        (item for item in results if item["min_strict"] >= 20.),
        key=lambda item: (item["gain_2024"], item["min_strict"]), reverse=True,
    )
    output = {
        "baseline": baseline, "calibration_specs": calibration_specs,
        "robust": robust, "min20_gain": positive,
    }
    path = ROOT / "research/v25_f_calibration_addition.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": {
            "min_strict": baseline["min_strict"], "gain_2024": baseline["gain_2024"],
        },
        "calibration_specs": calibration_specs,
        "robust": robust[:10], "min20_gain": positive[:10],
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
