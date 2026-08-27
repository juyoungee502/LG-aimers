"""Measure gain sacrificed by the v25 strict segment constraints.

The candidate library and all table directions are unchanged.  Only the set of
forward-transfer constraints is relaxed, which exposes whether an aggressive
submission has enough honest upside to justify its temporal risk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_v25_f_combined_portfolio import (
    candidate_name, direction,
)
from research_v25_f_pair_transfer import context_frame
from research_v25_f_portfolio import curve, exact_metrics, masks


ROOT = Path(__file__).resolve().parent


def optimize_gain(curves, objective_curve, cap, seeds):
    width = len(objective_curve[0])

    def score(value, weights):
        linear, quadratic = value
        return float(linear @ weights - weights @ quadratic @ weights)

    def objective(weights):
        return -score(objective_curve, weights) + .001 * float(weights @ weights)

    constraints = [
        {"type": "ineq", "fun": lambda weights, c=c: score(c, weights)}
        for c in curves.values()
    ]
    constraints.append({
        "type": "ineq", "fun": lambda weights: cap - weights.sum(),
    })
    starts = [np.zeros(width), *seeds]
    rng = np.random.default_rng(20260828 + int(cap * 100))
    for _ in range(5):
        value = rng.random(width)
        starts.append(value / value.sum() * min(cap * .6, 2.))
    results = []
    for start in starts:
        start = np.asarray(start, dtype=float).copy()
        if start.sum() > cap:
            start *= cap / start.sum()
        result = minimize(
            objective, start, method="SLSQP", bounds=[(0., 1.)] * width,
            constraints=constraints,
            options={"maxiter": 1600, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            results.append(result)
    if not results:
        raise RuntimeError(f"No frontier solution for cap={cap}")
    return min(results, key=lambda item: item.fun)


def selected_seed(payload, regime, candidates):
    target = (2.5, .05) if regime == "R" else (3., .05)
    solution = next(
        item for item in payload["solutions"]
        if item["cap"] == target[0] and item["rounding"] == target[1]
    )
    return np.asarray([
        solution["weights"].get(candidate_name(candidate), 0.)
        for candidate in candidates
    ])


def constraint_tiers(all_curves):
    return {
        "transfer_all": {
            key: value for key, value in all_curves.items()
            if key.endswith("/all") and not key.startswith("2024/")
        },
        "transfer_halves": {
            key: value for key, value in all_curves.items()
            if not key.startswith("2024/")
            and key.rsplit("/", 1)[-1] in ("all", "half_1", "half_2")
        },
        "transfer_quarters": {
            key: value for key, value in all_curves.items()
            if not key.startswith("2024/")
            and not key.rsplit("/", 1)[-1].startswith("month_")
        },
        "strict_61": dict(all_curves),
    }


def frontier(regime, rows, numeric, categorical, context, y, base, year):
    payload = json.loads((
        ROOT / f"research/v25_{regime.lower()}_combined_portfolio.json"
    ).read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    active = rows["game_type"].eq(regime).to_numpy()
    indices = {
        value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)
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
    residual = y - base
    blocks = {}
    all_curves = {}
    for label, source, valid in transfers:
        directions = np.stack([
            direction(
                candidate, numeric, categorical, context, source, valid, residual,
            ) for candidate in candidates
        ])
        block_masks = masks(rows.iloc[valid].reset_index(drop=True), label)
        blocks[label] = {
            "y": y[valid], "base": base[valid], "directions": directions,
            "masks": block_masks,
        }
        for name, mask in block_masks.items():
            all_curves[name] = curve(y[valid], base[valid], directions, mask)
    source, valid = indices[2023], indices[2024]
    full_directions = np.stack([
        direction(
            candidate, numeric, categorical, context, source, valid, residual,
        ) for candidate in candidates
    ])
    full_masks = masks(rows.iloc[valid].reset_index(drop=True), "2024")
    for name, mask in full_masks.items():
        all_curves[name] = curve(y[valid], base[valid], full_directions, mask)
    objective_curve = all_curves["2024/all"]
    seed = selected_seed(payload, regime, candidates)

    reports = []
    for tier, constraints in constraint_tiers(all_curves).items():
        for cap in ((2.5, 4., 6.) if regime == "R" else (3., 4., 6.)):
            try:
                result = optimize_gain(constraints, objective_curve, cap, [seed])
            except RuntimeError as error:
                reports.append({
                    "tier": tier, "cap": cap, "rounding": None,
                    "error": str(error), "gain_2024": -1e9,
                })
                continue
            for rounding in (None, .05):
                weights = result.x.copy()
                if rounding is not None:
                    weights = np.round(weights / rounding) * rounding
                    # Rounding can slightly violate the cap.  Scaling preserves
                    # the learned direction while restoring the complexity gate.
                    if weights.sum() > cap:
                        weights *= cap / weights.sum()
                exact = {}
                for block in blocks.values():
                    exact.update(exact_metrics(
                        block["y"], block["base"], block["directions"], weights,
                        block["masks"],
                    ))
                exact.update(exact_metrics(
                    y[valid], base[valid], full_directions, weights, full_masks,
                ))
                reports.append({
                    "tier": tier, "cap": cap, "rounding": rounding,
                    "gain_2024": exact["2024/all"],
                    "min_all_transfer": min(
                        value for key, value in exact.items()
                        if key.endswith("/all") and not key.startswith("2024/")
                    ),
                    "min_transfer_half": min(
                        value for key, value in exact.items()
                        if not key.startswith("2024/")
                        and key.rsplit("/", 1)[-1] in ("all", "half_1", "half_2")
                    ),
                    "min_transfer_quarter": min(
                        value for key, value in exact.items()
                        if not key.startswith("2024/")
                        and not key.rsplit("/", 1)[-1].startswith("month_")
                    ),
                    "min_strict": min(exact.values()),
                    "weight_sum": float(weights.sum()),
                    "weights": {
                        candidate_name(candidate): float(weight)
                        for candidate, weight in zip(candidates, weights)
                        if weight > 1e-7
                    },
                    "metrics": exact,
                })
    reports.sort(key=lambda item: item["gain_2024"], reverse=True)
    return reports


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
    year = oof["season"].astype(int)
    output = {}
    for regime in ("R", "F"):
        output[regime] = frontier(
            regime, rows, numeric, categorical, context, y, base, year,
        )
        print(json.dumps({
            "regime": regime, "top": [
                {
                    key: item.get(key) for key in (
                        "tier", "cap", "rounding", "gain_2024",
                        "min_all_transfer", "min_transfer_half",
                        "min_transfer_quarter", "min_strict", "weight_sum",
                        "error",
                    )
                }
                for item in output[regime][:20]
            ],
        }, indent=2), flush=True)
    path = ROOT / "research/v26_constraint_frontier.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
