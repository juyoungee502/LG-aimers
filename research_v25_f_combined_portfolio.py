"""Jointly optimize one- and two-dimensional F residual effects.

The two-dimensional screen is intentionally not trusted on its own: candidate
tables enter the same 61 chronological/monthly constraints used by the robust
one-dimensional portfolio.  This script therefore measures their *marginal*
value after the existing safe portfolio instead of selecting a visually strong
full-year result.
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
from research_v25_f_pair_transfer import context_frame, pair_codes
from research_v25_f_portfolio import curve, exact_metrics, masks
from research_v24_exhaustive_transfer import (
    encode_categorical, encode_numeric, table_direction,
)


ROOT = Path(__file__).resolve().parent
MAX_PAIR_CANDIDATES = 24
CORRELATION_LIMIT = .985


def one_d_direction(spec, numeric, categorical, source, valid, residual):
    values = numeric[spec["column"]] if spec["kind"] == "numeric" \
        else categorical[spec["column"]]
    encoded = (
        encode_numeric(values.iloc[source], values.iloc[valid], int(spec["bins"]))
        if spec["kind"] == "numeric"
        else encode_categorical(values.iloc[source], values.iloc[valid])
    )
    if encoded is None:
        raise ValueError(f"Could not encode {spec['column']}")
    return table_direction(
        encoded[0], encoded[1], residual[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def pair_direction(spec, numeric, context, source, valid, residual):
    encoded = pair_codes(
        numeric[spec["column"]].iloc[source], numeric[spec["column"]].iloc[valid],
        context[spec["context"]].iloc[source], context[spec["context"]].iloc[valid],
        int(spec["bins"]),
    )
    if encoded is None:
        raise ValueError(f"Could not encode {spec['column']} x {spec['context']}")
    return table_direction(
        encoded[0], encoded[1], residual[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def candidate_name(candidate):
    spec = candidate["spec"]
    if candidate["type"] == "one_d":
        return f"1d:{spec['kind']}:{spec['column']}"
    return f"pair:{spec['column']}:{spec['context']}"


def direction(candidate, numeric, categorical, context, source, valid, residual):
    if candidate["type"] == "one_d":
        return one_d_direction(
            candidate["spec"], numeric, categorical, source, valid, residual,
        )
    return pair_direction(
        candidate["spec"], numeric, context, source, valid, residual,
    )


def segment_curves(blocks, y, base, full_directions, full_masks, valid):
    strict = {}
    for block in blocks.values():
        for name, active in block["masks"].items():
            strict[name] = curve(
                block["y"], block["base"], block["directions"], active,
            )
    for name, active in full_masks.items():
        strict[name] = curve(y[valid], base[valid], full_directions, active)
    return strict


def optimize(curves, cap, seeds):
    width = len(next(iter(curves.values()))[0])

    def score(curve_value, weights):
        linear, quadratic = curve_value
        return float(linear @ weights - weights @ quadratic @ weights)

    def objective(value):
        return -value[-1] + .002 * float(value[:-1] @ value[:-1])

    constraints = [
        {"type": "ineq", "fun": lambda value, c=c: score(c, value[:-1]) - value[-1]}
        for c in curves.values()
    ]
    constraints.append({"type": "ineq", "fun": lambda value: cap - value[:-1].sum()})
    rng = np.random.default_rng(20260828 + int(cap * 100))
    initial = [np.zeros(width), *seeds]
    for _ in range(8):
        value = rng.random(width)
        initial.append(value / value.sum() * min(cap * .75, 1.75))
    results = []
    for weights in initial:
        weights = np.asarray(weights, dtype=float).copy()
        if weights.sum() > cap:
            weights *= cap / weights.sum()
        t0 = min(score(item, weights) for item in curves.values()) - .1
        result = minimize(
            objective, np.r_[weights, t0], method="SLSQP",
            bounds=[(0., 1.)] * width + [(-500., 500.)],
            constraints=constraints,
            options={"maxiter": 1800, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            results.append(result)
    if not results:
        raise RuntimeError(f"No optimizer result for cap={cap}")
    return min(results, key=lambda item: item.fun)


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

    portfolio = json.loads(
        (ROOT / "research/v25_f_portfolio.json").read_text(encoding="utf-8")
    )
    candidates = [{"type": "one_d", "spec": spec} for spec in portfolio["candidates"]]
    baseline_solution = next(
        item for item in portfolio["solutions"]
        if item["gate"] == "strict" and item["cap"] == 2.0 and item["rounding"] == .05
    )
    baseline_weights = {
        f"1d:{spec['kind']}:{spec['column']}": baseline_solution["weights"].get(spec["column"], 0.)
        for spec in portfolio["candidates"]
    }

    # Fit direction vectors before pair selection so redundant 2-D tables do
    # not get multiple chances merely because their labels differ.
    vectors = []
    transfer_values = []
    for candidate in candidates:
        values = {
            label: direction(candidate, numeric, categorical, context, source, valid, residual)
            for label, source, valid in transfers
        }
        transfer_values.append(values)
        vectors.append(np.concatenate([values[label] for label, _s, _v in transfers]))

    pair_payload = json.loads(
        (ROOT / "research/v25_f_pair_transfer.json").read_text(encoding="utf-8")
    )
    unique_pairs = {}
    for spec in pair_payload["top"]:
        unique_pairs.setdefault((spec["column"], spec["context"]), spec)
    for spec in unique_pairs.values():
        candidate = {"type": "pair", "spec": spec}
        values = {
            label: direction(candidate, numeric, categorical, context, source, valid, residual)
            for label, source, valid in transfers
        }
        vector = np.concatenate([values[label] for label, _s, _v in transfers])
        if any(abs(np.corrcoef(vector, old)[0, 1]) > CORRELATION_LIMIT for old in vectors):
            continue
        candidates.append(candidate)
        transfer_values.append(values)
        vectors.append(vector)
        if sum(item["type"] == "pair" for item in candidates) >= MAX_PAIR_CANDIDATES:
            break
    print(
        f"combined candidates={len(candidates)} "
        f"pairs={sum(item['type'] == 'pair' for item in candidates)}", flush=True,
    )

    blocks = {}
    for label, _source, valid in transfers:
        directions = np.stack([values[label] for values in transfer_values])
        blocks[label] = {
            "y": y[valid], "base": base[valid], "directions": directions,
            "masks": masks(rows.iloc[valid].reset_index(drop=True), label),
        }
    source, valid = indices[2023], indices[2024]
    full_directions = np.stack([
        direction(candidate, numeric, categorical, context, source, valid, residual)
        for candidate in candidates
    ])
    full_masks = masks(rows.iloc[valid].reset_index(drop=True), "2024")
    curves = segment_curves(blocks, y, base, full_directions, full_masks, valid)

    seed = np.array([baseline_weights.get(candidate_name(item), 0.) for item in candidates])
    solutions = []
    for cap in (2., 3., 4.):
        result = optimize(curves, cap, [seed])
        for rounding in (None, .025, .05, .10):
            weights = result.x[:-1].copy()
            if rounding is not None:
                weights = np.round(weights / rounding) * rounding
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
            pair_weight = float(sum(
                weight for candidate, weight in zip(candidates, weights)
                if candidate["type"] == "pair"
            ))
            solutions.append({
                "cap": cap, "rounding": rounding,
                "optimizer_t": float(result.x[-1]),
                "weights": {
                    candidate_name(candidate): float(weight)
                    for candidate, weight in zip(candidates, weights) if weight > 1e-7
                },
                "weight_sum": float(weights.sum()), "pair_weight": pair_weight,
                "min_strict": float(min(exact.values())),
                "gain_2024": float(exact["2024/all"]), "metrics": exact,
            })
    solutions.sort(key=lambda item: (item["min_strict"], item["gain_2024"]), reverse=True)
    output = {
        "candidates": candidates, "constraint_count": len(curves),
        "baseline_solution": baseline_solution, "solutions": solutions,
    }
    path = ROOT / "research/v25_f_combined_portfolio.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "constraint_count": len(curves), "candidates": candidates,
        "top": solutions[:15],
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
