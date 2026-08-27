"""Re-optimize the v26 portfolio with exposure-aware numeric 2-D tables."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy.optimize import minimize

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v25_f_calibration_addition import probability_direction
from research_v25_f_combined_portfolio import candidate_name, direction
from research_v25_f_pair_transfer import context_frame
from research_v25_f_portfolio import curve, exact_metrics, masks
from research_v26_constraint_frontier import optimize_gain, selected_seed
from research_v27_numeric_2d import numeric_2d_codes
from research_v24_exhaustive_transfer import table_direction


ROOT = Path(__file__).resolve().parent
MAX_2D = {"R": 20, "F": 10}
CORRELATION_LIMIT = .997
warnings.filterwarnings("ignore", category=PerformanceWarning)


def optimize_extended(curves, objective_curve, cap, seeds, floor):
    """Solve the convex Brier frontier with analytical SLSQP gradients.

    The F curves are larger and more collinear than R.  Analytical Jacobians
    avoid the singular finite-difference exit seen when numeric 2-D axes are
    appended to the original pool.  A numerically feasible SLSQP endpoint is
    accepted even if SciPy reports a line-search warning.
    """
    linear_objective, quadratic_objective = objective_curve
    quadratic_objective = .5 * (quadratic_objective + quadratic_objective.T)
    width = len(linear_objective)

    def score(value, weights):
        linear, quadratic = value
        return float(linear @ weights - weights @ quadratic @ weights)

    def objective(weights):
        return float(
            -linear_objective @ weights
            + weights @ quadratic_objective @ weights
            + .001 * weights @ weights
        )

    def objective_jac(weights):
        return (
            -linear_objective + 2. * quadratic_objective @ weights
            + .002 * weights
        )

    constraints = []
    for linear, quadratic in curves.values():
        quadratic = .5 * (quadratic + quadratic.T)
        constraints.append({
            "type": "ineq",
            "fun": lambda weights, a=linear, q=quadratic: float(
                a @ weights - weights @ q @ weights - floor
            ),
            "jac": lambda weights, a=linear, q=quadratic: a - 2. * q @ weights,
        })
    constraints.append({
        "type": "ineq",
        "fun": lambda weights: float(cap - weights.sum()),
        "jac": lambda weights: -np.ones_like(weights),
    })
    starts = [np.zeros(width), *seeds]
    for seed in seeds:
        starts.extend((.75 * np.asarray(seed), .9 * np.asarray(seed)))
    rng = np.random.default_rng(20260831 + int(cap * 10 + floor))
    for _ in range(3):
        value = rng.random(width)
        starts.append(value / value.sum() * min(cap * .6, 2.))
    results = []
    messages = []
    for start in starts:
        start = np.clip(np.asarray(start, dtype=float), 0., 1.)
        if start.sum() > cap:
            start *= cap / start.sum()
        result = minimize(
            objective, start, jac=objective_jac, method="SLSQP",
            bounds=[(0., 1.)] * width, constraints=constraints,
            options={"maxiter": 2500, "ftol": 1e-10, "disp": False},
        )
        minimum = min(score(item, result.x) for item in curves.values())
        feasible = minimum >= floor - 1e-5 and result.x.sum() <= cap + 1e-6
        if feasible:
            results.append(result)
        else:
            messages.append((result.message, minimum, float(result.x.sum())))
    if not results:
        raise RuntimeError(
            f"No feasible extended frontier cap={cap} floor={floor}: {messages[:3]}"
        )
    return min(results, key=lambda item: item.fun)


def numeric_2d_direction(spec, numeric, source, valid, residual):
    codes = numeric_2d_codes(
        numeric[spec["x"]].iloc[source], numeric[spec["x"]].iloc[valid],
        numeric[spec["exposure"]].iloc[source],
        numeric[spec["exposure"]].iloc[valid],
        int(spec["bins_x"]), int(spec["bins_exposure"]),
    )
    if codes is None:
        raise ValueError(f"Could not encode {spec['x']} x {spec['exposure']}")
    return table_direction(
        codes[0], codes[1], residual[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def name(candidate):
    if candidate["type"] in ("one_d", "pair"):
        return candidate_name(candidate)
    if candidate["type"] == "probability_pair":
        spec = candidate["spec"]
        return f"probability:{spec['context']}:{spec['bins']}:{spec['shrink']}"
    spec = candidate["spec"]
    return (
        f"numeric2d:{spec['x']}:{spec['exposure']}:"
        f"{spec['bins_x']}x{spec['bins_exposure']}:"
        f"{spec['shrink']}:{spec['scale']}"
    )


def candidate_direction(
    candidate, numeric, categorical, context, base, source, valid, residual,
):
    if candidate["type"] in ("one_d", "pair"):
        return direction(
            candidate, numeric, categorical, context, source, valid, residual,
        )
    if candidate["type"] == "probability_pair":
        return probability_direction(
            candidate["spec"], base, context, source, valid, residual,
        )
    return numeric_2d_direction(
        candidate["spec"], numeric, source, valid, residual,
    )


def extra_candidates(regime):
    payload = json.loads(
        (ROOT / "research/v27_numeric_2d.json").read_text(encoding="utf-8")
    )[regime]
    ordered = []
    seen = set()
    rankings = (
        payload["robust"],
        payload["positive_transfer"],
        sorted(
            payload["positive_transfer"],
            key=lambda item: item["min_all_segment_absolute"], reverse=True,
        ),
    )
    for ranking in rankings:
        for item in ranking:
            key = (
                item["x"], item["exposure"], item["bins_x"],
                item["bins_exposure"], item["shrink"], item["scale"],
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append({
                "type": "numeric_2d",
                "spec": {field: item[field] for field in (
                    "x", "exposure", "bins_x", "bins_exposure", "shrink", "scale",
                )},
            })
            if len(ordered) >= MAX_2D[regime]:
                return ordered
    return ordered


def regime_frontier(regime, rows, numeric, categorical, context, y, base, year):
    payload = json.loads((
        ROOT / f"research/v25_{regime.lower()}_combined_portfolio.json"
    ).read_text(encoding="utf-8"))
    original = payload["candidates"]
    candidates = list(original)
    if regime == "F":
        candidates.append({
            "type": "probability_pair",
            "spec": {
                "context": "inning_bucket", "bins": 8,
                "shrink": 100., "scale": .1,
            },
        })
    additions = extra_candidates(regime)
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

    # Remove only almost-identical new axes; preserve the audited original pool.
    original_vectors = []
    for candidate in candidates:
        original_vectors.append(np.concatenate([
            candidate_direction(
                candidate, numeric, categorical, context, base,
                source, valid, residual,
            )
            for _label, source, valid in transfers
        ]))
    vectors = list(original_vectors)
    for candidate in additions:
        vector = np.concatenate([
            candidate_direction(
                candidate, numeric, categorical, context, base,
                source, valid, residual,
            )
            for _label, source, valid in transfers
        ])
        if any(abs(np.corrcoef(vector, old)[0, 1]) > CORRELATION_LIMIT for old in vectors):
            continue
        candidates.append(candidate)
        vectors.append(vector)
    print(
        f"{regime}: original={len(original)} total={len(candidates)} "
        f"numeric2d={sum(c['type'] == 'numeric_2d' for c in candidates)}",
        flush=True,
    )

    blocks = {}
    all_curves = {}
    for label, source, valid in transfers:
        directions = np.stack([
            candidate_direction(
                candidate, numeric, categorical, context, base,
                source, valid, residual,
            )
            for candidate in candidates
        ])
        block_masks = masks(rows.iloc[valid].reset_index(drop=True), label)
        blocks[label] = {
            "y": y[valid], "base": base[valid],
            "directions": directions, "masks": block_masks,
        }
        for key, mask in block_masks.items():
            all_curves[key] = curve(y[valid], base[valid], directions, mask)
    source, valid = indices[2023], indices[2024]
    full_directions = np.stack([
        candidate_direction(
            candidate, numeric, categorical, context, base,
            source, valid, residual,
        )
        for candidate in candidates
    ])
    full_masks = masks(rows.iloc[valid].reset_index(drop=True), "2024")
    for key, mask in full_masks.items():
        all_curves[key] = curve(y[valid], base[valid], full_directions, mask)

    seed = np.zeros(len(candidates), dtype=float)
    frontier = json.loads((
        ROOT / "research/v26_constraint_frontier.json"
    ).read_text(encoding="utf-8"))[regime]
    if regime == "R":
        baseline = next(
            item for item in frontier
            if item["tier"] == "strict_floor_5" and item["cap"] == 6.
            and item["rounding"] is None
        )
    else:
        baseline = next(
            item for item in frontier
            if item["tier"] == "strict_floor_15" and item["cap"] == 6.
            and item["rounding"] == .05
        )
    seed[:len(original)] = np.asarray([
        baseline["weights"].get(candidate_name(candidate), 0.)
        for candidate in original
    ])
    if regime == "F":
        seed[len(original)] = .25
    floors = (0., 3., 5.) if regime == "R" else (0., 5., 10.)
    reports = []
    for floor in floors:
        for cap in (6., 8.):
            result = optimize_extended(
                all_curves, all_curves["2024/all"], cap, [seed], floor=floor,
            )
            for rounding in (None, .025, .05):
                weights = result.x.copy()
                if rounding is not None:
                    weights = np.round(weights / rounding) * rounding
                if weights.sum() > cap:
                    weights *= cap / weights.sum()
                exact = {}
                for block in blocks.values():
                    exact.update(exact_metrics(
                        block["y"], block["base"], block["directions"],
                        weights, block["masks"],
                    ))
                exact.update(exact_metrics(
                    y[valid], base[valid], full_directions, weights, full_masks,
                ))
                reports.append({
                    "floor": floor, "cap": cap, "rounding": rounding,
                    "gain_2024": exact["2024/all"],
                    "min_strict": min(exact.values()),
                    "weight_sum": float(weights.sum()),
                    "weights": {
                        name(candidate): float(weight)
                        for candidate, weight in zip(candidates, weights)
                        if weight > 1e-7
                    },
                    "metrics": exact,
                })
            print(f"{regime}: optimized floor={floor} cap={cap}", flush=True)
    reports.sort(key=lambda item: (item["gain_2024"], item["min_strict"]), reverse=True)
    return {"candidates": candidates, "reports": reports}


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(raw, target_series)
    numeric_all = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
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
        output[regime] = regime_frontier(
            regime, rows, numeric, categorical, context, y, base, year,
        )
        print(json.dumps({
            "regime": regime,
            "top": [
                {key: item[key] for key in (
                    "floor", "cap", "rounding", "gain_2024",
                    "min_strict", "weight_sum",
                )}
                for item in output[regime]["reports"][:20]
            ],
        }, indent=2), flush=True)
    path = ROOT / "research/v27_extended_frontier.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
