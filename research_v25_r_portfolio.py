"""Optimize a chronologically constrained regular-season residual portfolio."""
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
from research_v25_f_portfolio import curve, exact_metrics, masks
from research_v24_exhaustive_transfer import (
    encode_categorical, encode_numeric, table_direction,
)


ROOT = Path(__file__).resolve().parent
BASELINE_FEATURES = (
    "asof_pitcher_success_rate",
    "pitcher_middle_season_rate",
    "pitcher_success_x_runners",
    "asof_pitcher_prev5_game_success_rate",
)
MAX_CANDIDATES = 32


def correction(spec, numeric, categorical, source, valid, residual):
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


def score(curve_value, weights):
    linear, quadratic = curve_value
    return float(linear @ weights - weights @ quadratic @ weights)


def optimize(curves, cap, seeds):
    width = len(next(iter(curves.values()))[0])

    def objective(value):
        return -value[-1] + .002 * float(value[:-1] @ value[:-1])

    constraints = [
        {"type": "ineq", "fun": lambda value, c=c: score(c, value[:-1]) - value[-1]}
        for c in curves.values()
    ]
    constraints.append({"type": "ineq", "fun": lambda value: cap - value[:-1].sum()})
    rng = np.random.default_rng(20260829 + int(cap * 100))
    initial = [np.zeros(width), *seeds]
    for _ in range(10):
        value = rng.random(width)
        initial.append(value / value.sum() * min(cap * .7, 1.5))
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
        raise RuntimeError(f"No solution for cap={cap}")
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
    rows = raw.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    residual = y - base
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

    payload = json.loads(
        (ROOT / "research/v24_exhaustive_transfer.json").read_text(encoding="utf-8")
    )
    strongest = {}
    for spec in payload["top"]:
        strongest.setdefault((spec["kind"], spec["column"]), spec)
    ordered = []
    for name in BASELINE_FEATURES:
        match = next(spec for spec in strongest.values() if spec["column"] == name)
        ordered.append(match)
    ordered.extend(
        spec for spec in strongest.values()
        if spec["column"] not in BASELINE_FEATURES
    )

    chosen = []
    vectors = []
    transfer_directions = []
    for spec in ordered:
        values = {
            label: correction(spec, numeric, categorical, source, valid, residual)
            for label, source, valid in transfers
        }
        vector = np.concatenate([values[label] for label, _s, _v in transfers])
        if any(abs(np.corrcoef(vector, old)[0, 1]) > .995 for old in vectors):
            continue
        chosen.append(spec)
        vectors.append(vector)
        transfer_directions.append(values)
        if len(chosen) >= MAX_CANDIDATES:
            break
    print(f"R portfolio candidates={len(chosen)}", flush=True)

    blocks = {}
    for label, _source, valid in transfers:
        blocks[label] = {
            "y": y[valid], "base": base[valid],
            "directions": np.stack([values[label] for values in transfer_directions]),
            "masks": masks(rows.iloc[valid].reset_index(drop=True), label),
        }
    source, valid = indices[2023], indices[2024]
    full_directions = np.stack([
        correction(spec, numeric, categorical, source, valid, residual)
        for spec in chosen
    ])
    full_masks = masks(rows.iloc[valid].reset_index(drop=True), "2024")

    strict_curves = {}
    core_curves = {}
    for label, block in blocks.items():
        for name, active in block["masks"].items():
            item = curve(block["y"], block["base"], block["directions"], active)
            strict_curves[name] = item
            if name.endswith("/all"):
                core_curves[name] = item
    for name, active in full_masks.items():
        item = curve(y[valid], base[valid], full_directions, active)
        strict_curves[name] = item
        core_curves[name] = item

    robust = json.loads(
        (ROOT / "research/v25_robust_mix.json").read_text(encoding="utf-8")
    )["top"][0]["weights"]
    seed = np.array([robust.get(spec["column"], 0.) for spec in chosen])
    solutions = []
    for gate, curves in (("core", core_curves), ("strict", strict_curves)):
        for cap in (1.5, 2., 3.):
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
                solutions.append({
                    "gate": gate, "cap": cap, "rounding": rounding,
                    "optimizer_t": float(result.x[-1]),
                    "weights": {
                        spec["column"]: float(weight)
                        for spec, weight in zip(chosen, weights) if weight > 1e-7
                    },
                    "weight_sum": float(weights.sum()), "metrics": exact,
                    "min_strict": float(min(exact.values())),
                    "min_core": float(min(exact[name] for name in curves)),
                    "gain_2024": float(exact["2024/all"]),
                })
    solutions.sort(
        key=lambda item: (
            item["gate"] == "strict", item["min_strict"], item["min_core"],
            item["gain_2024"],
        ), reverse=True,
    )
    output = {
        "candidates": chosen, "strict_constraint_count": len(strict_curves),
        "core_constraint_count": len(core_curves), "solutions": solutions,
    }
    path = ROOT / "research/v25_r_portfolio.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    strict_top = sorted(
        (item for item in solutions if item["gate"] == "strict"),
        key=lambda item: (item["min_strict"], item["gain_2024"]), reverse=True,
    )[:12]
    core_top = sorted(
        (item for item in solutions if item["gate"] == "core"),
        key=lambda item: (item["min_core"], item["min_strict"], item["gain_2024"]),
        reverse=True,
    )[:12]
    print(json.dumps({
        "strict_constraints": len(strict_curves), "core_constraints": len(core_curves),
        "candidates": chosen, "strict_top": strict_top, "core_top": core_top,
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
