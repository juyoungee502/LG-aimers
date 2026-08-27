"""Audit probability-table additions after the v26 F Pareto portfolio."""
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
from research_v25_f_calibration_addition import probability_direction
from research_v25_f_combined_portfolio import candidate_name, direction
from research_v25_f_pair_transfer import context_frame
from research_v25_f_portfolio import exact_metrics, masks


ROOT = Path(__file__).resolve().parent
GRID = (0., .25, .5, .75, 1.)


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    numeric_all = engineer_features(
        raw, *bases, global_prior=float(target_series.mean()),
    )
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
        np.flatnonzero(seasons == value) for value in (2023, 2024)
    ])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    rows = raw.iloc[positions].reset_index(drop=True)
    numeric = numeric_all.iloc[positions].reset_index(drop=True)
    categorical = categorical_all.iloc[positions].reset_index(drop=True)
    context = context_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    residual = y - base
    year = oof["season"].astype(int)
    futures = rows["game_type"].eq("F").to_numpy()
    indices = {
        value: np.flatnonzero(futures & (year == value)) for value in (2023, 2024)
    }
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

    combined = json.loads((
        ROOT / "research/v25_f_combined_portfolio.json"
    ).read_text(encoding="utf-8"))
    candidates = combined["candidates"]
    frontier = json.loads((
        ROOT / "research/v26_constraint_frontier.json"
    ).read_text(encoding="utf-8"))["F"]
    baseline = next(
        item for item in frontier
        if item["tier"] == "strict_floor_15" and item["cap"] == 6.
        and item["rounding"] == .05
    )
    weights = np.asarray([
        baseline["weights"].get(candidate_name(candidate), 0.)
        for candidate in candidates
    ])
    probability = json.loads((
        ROOT / "research/v25_probability_tables.json"
    ).read_text(encoding="utf-8"))["F"]["top"]
    unique = {}
    for spec in probability:
        unique.setdefault(spec["context"], spec)
    calibration_specs = list(unique.values())

    blocks = {}
    for label, source, valid in transfers:
        base_directions = np.stack([
            direction(
                candidate, numeric, categorical, context, source, valid, residual,
            ) for candidate in candidates
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
        direction(
            candidate, numeric, categorical, context, source, valid, residual,
        ) for candidate in candidates
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
    safe = sorted(
        (item for item in results if item["min_strict"] >= 10.),
        key=lambda item: (item["gain_2024"], item["min_strict"]), reverse=True,
    )
    robust = sorted(
        results, key=lambda item: (item["min_strict"], item["gain_2024"]),
        reverse=True,
    )
    output = {
        "baseline": baseline, "calibration_specs": calibration_specs,
        "safe": safe, "robust": robust,
    }
    path = ROOT / "research/v26_calibration_addition.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": {
            "gain_2024": baseline["gain_2024"],
            "min_strict": baseline["min_strict"],
        },
        "safe": safe[:10], "robust": robust[:10],
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
