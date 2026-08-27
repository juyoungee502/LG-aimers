"""Build a robust portfolio from post-break F-regime residual effects."""
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
from research_inferred_pitch_priors import bss
from research_v24_exhaustive_transfer import (
    encode_categorical, encode_numeric, table_direction,
)


ROOT = Path(__file__).resolve().parent
MAX_CANDIDATES = 24


def encode(spec, values, source, valid):
    if spec["kind"] == "numeric":
        return encode_numeric(values.iloc[source], values.iloc[valid], int(spec["bins"]))
    return encode_categorical(values.iloc[source], values.iloc[valid])


def correction(spec, numeric, categorical, source, valid, residual):
    values = numeric[spec["column"]] if spec["kind"] == "numeric" \
        else categorical[spec["column"]]
    codes = encode(spec, values, source, valid)
    return table_direction(
        codes[0], codes[1], residual[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def masks(rows, prefix):
    position = np.arange(len(rows))
    result = {
        f"{prefix}/all": np.ones(len(rows), dtype=bool),
        f"{prefix}/half_1": position < len(rows) // 2,
        f"{prefix}/half_2": position >= len(rows) // 2,
        f"{prefix}/q1": position < len(rows) // 4,
        f"{prefix}/q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        f"{prefix}/q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        f"{prefix}/q4": position >= 3 * len(rows) // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        if active.sum() >= 40:
            result[f"{prefix}/month_{int(month)}"] = active
    return result


def curve(y, base, directions, active):
    uncertainty = float(y[active].mean() * (1. - y[active].mean()))
    residual = y[active] - base[active]
    matrix = directions[:, active]
    linear = 200000. * (matrix @ residual) / (active.sum() * uncertainty)
    quadratic = 100000. * (matrix @ matrix.T) / (active.sum() * uncertainty)
    return linear, quadratic


def gain(curve_value, weights):
    linear, quadratic = curve_value
    return float(linear @ weights - weights @ quadratic @ weights)


def optimize(curves, cap, starts=10):
    width = len(next(iter(curves.values()))[0])

    def objective(value):
        # Tiny complexity penalty chooses the simpler point on a flat frontier.
        return -value[-1] + .002 * float(value[:-1] @ value[:-1])

    constraints = [
        {"type": "ineq", "fun": lambda value, c=c: gain(c, value[:-1]) - value[-1]}
        for c in curves.values()
    ]
    constraints.append({"type": "ineq", "fun": lambda value: cap - value[:-1].sum()})
    rng = np.random.default_rng(20260827 + int(cap * 10))
    results = []
    initial_weights = [
        np.zeros(width),
        np.r_[.25, np.zeros(width - 1)],
        *[
            (lambda v: v / max(v.sum(), 1e-9) * min(cap * .7, 1.5))(rng.random(width))
            for _ in range(starts - 2)
        ],
    ]
    for weights in initial_weights:
        initial_t = min(gain(value, weights) for value in curves.values()) - .1
        result = minimize(
            objective, np.r_[weights, initial_t], method="SLSQP",
            bounds=[(0., 1.)] * width + [(-500., 500.)],
            constraints=constraints,
            options={"maxiter": 1200, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            results.append(result)
    if not results:
        raise RuntimeError(f"No optimization result for cap={cap}")
    return min(results, key=lambda result: result.fun)


def exact_metrics(y, base, directions, weights, segment_masks):
    correction_value = np.tensordot(weights, directions, axes=1)
    candidate = np.clip(base + correction_value, .005, .995)
    result = {}
    for name, active in segment_masks.items():
        result[name] = bss(y[active], candidate[active]) - bss(y[active], base[active])
    return result


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

    screened = json.loads(
        (ROOT / "research/v25_f_exhaustive_transfer.json").read_text(encoding="utf-8")
    )["top"]
    # One strongest setting per feature, then remove virtually identical
    # directions.  This keeps the constrained optimization low variance.
    unique = {}
    for item in screened:
        unique.setdefault((item["kind"], item["column"]), item)
    preliminary = list(unique.values())
    transfer_directions = {}
    chosen = []
    flattened = []
    for spec in preliminary:
        values = {
            label: correction(spec, numeric, categorical, source, valid, residual)
            for label, source, valid in transfers
        }
        vector = np.concatenate([values[label] for label, _source, _valid in transfers])
        if any(abs(np.corrcoef(vector, old)[0, 1]) > .995 for old in flattened):
            continue
        chosen.append(spec)
        flattened.append(vector)
        transfer_directions[(spec["kind"], spec["column"])] = values
        if len(chosen) >= MAX_CANDIDATES:
            break
    print(f"portfolio candidates={len(chosen)}", flush=True)

    blocks = {}
    for label, source, valid in transfers:
        directions = np.stack([
            transfer_directions[(spec["kind"], spec["column"])][label]
            for spec in chosen
        ])
        block_masks = masks(rows.iloc[valid].reset_index(drop=True), label)
        blocks[label] = {
            "y": y[valid], "base": base[valid], "directions": directions,
            "masks": block_masks,
        }

    source, valid = indices[2023], indices[2024]
    full_directions = np.stack([
        correction(spec, numeric, categorical, source, valid, residual)
        for spec in chosen
    ])
    full_masks = masks(rows.iloc[valid].reset_index(drop=True), "2024")

    coarse_curves = {}
    strict_curves = {}
    for label, block in blocks.items():
        for name, active in block["masks"].items():
            value = curve(block["y"], block["base"], block["directions"], active)
            strict_curves[name] = value
            if name.endswith("/all"):
                coarse_curves[name] = value
    for name, active in full_masks.items():
        value = curve(y[valid], base[valid], full_directions, active)
        strict_curves[name] = value
        coarse_curves[name] = value

    solutions = []
    for gate_name, curves in (("coarse", coarse_curves), ("strict", strict_curves)):
        for cap in (1., 2., 3., 4.):
            result = optimize(curves, cap)
            raw_weights = result.x[:-1]
            for rounding in (None, .05, .10):
                weights = raw_weights if rounding is None else np.round(raw_weights / rounding) * rounding
                if weights.sum() > cap + 1e-8:
                    weights *= cap / weights.sum()
                exact = {}
                for block in blocks.values():
                    exact.update(exact_metrics(
                        block["y"], block["base"], block["directions"], weights,
                        block["masks"],
                    ))
                exact.update(exact_metrics(y[valid], base[valid], full_directions, weights, full_masks))
                solutions.append({
                    "gate": gate_name, "cap": cap,
                    "rounding": rounding, "optimizer_t": float(result.x[-1]),
                    "weights": {
                        spec["column"]: float(weight)
                        for spec, weight in zip(chosen, weights) if weight > 1e-6
                    },
                    "weight_sum": float(weights.sum()), "metrics": exact,
                    "min_strict": min(exact.values()),
                    "min_coarse": min(
                        value for name, value in exact.items()
                        if name in coarse_curves
                    ),
                    "gain_2024": exact["2024/all"],
                })
    solutions.sort(
        key=lambda item: (item["min_strict"], item["min_coarse"], item["gain_2024"]),
        reverse=True,
    )
    output = {
        "candidates": chosen, "coarse_constraint_count": len(coarse_curves),
        "strict_constraint_count": len(strict_curves), "solutions": solutions,
    }
    path = ROOT / "research/v25_f_portfolio.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidates": chosen, "coarse_constraint_count": len(coarse_curves),
        "strict_constraint_count": len(strict_curves), "top": solutions[:20],
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
