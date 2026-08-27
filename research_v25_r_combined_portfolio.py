"""Joint robust portfolio for regular-season one- and two-dimensional effects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_v25_f_combined_portfolio import (
    candidate_name, direction, optimize, segment_curves,
)
from research_v25_f_pair_transfer import context_frame
from research_v25_f_portfolio import exact_metrics, masks


ROOT = Path(__file__).resolve().parent
MAX_PAIR_CANDIDATES = 24


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

    portfolio = json.loads(
        (ROOT / "research/v25_r_portfolio.json").read_text(encoding="utf-8")
    )
    candidates = [{"type": "one_d", "spec": spec} for spec in portfolio["candidates"]]
    baseline = max(
        (
            item for item in portfolio["solutions"]
            if item["gate"] == "strict" and item["rounding"] == .025
        ),
        key=lambda item: (item["min_strict"], item["gain_2024"]),
    )
    baseline_weights = {
        f"1d:{spec['kind']}:{spec['column']}": baseline["weights"].get(spec["column"], 0.)
        for spec in portfolio["candidates"]
    }

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
        (ROOT / "research/v25_r_pair_transfer.json").read_text(encoding="utf-8")
    )
    unique = {}
    for spec in pair_payload["top"]:
        unique.setdefault((spec["column"], spec["context"]), spec)
    for spec in unique.values():
        candidate = {"type": "pair", "spec": spec}
        values = {
            label: direction(candidate, numeric, categorical, context, source, valid, residual)
            for label, source, valid in transfers
        }
        vector = np.concatenate([values[label] for label, _s, _v in transfers])
        if any(abs(np.corrcoef(vector, old)[0, 1]) > .985 for old in vectors):
            continue
        candidates.append(candidate)
        transfer_values.append(values)
        vectors.append(vector)
        if sum(item["type"] == "pair" for item in candidates) >= MAX_PAIR_CANDIDATES:
            break
    print(
        f"R combined candidates={len(candidates)} "
        f"pairs={sum(item['type'] == 'pair' for item in candidates)}", flush=True,
    )

    blocks = {}
    for label, _source, valid in transfers:
        blocks[label] = {
            "y": y[valid], "base": base[valid],
            "directions": np.stack([values[label] for values in transfer_values]),
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
    for cap in (2.5, 3., 4.):
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
                "cap": cap, "rounding": rounding,
                "optimizer_t": float(result.x[-1]),
                "weights": {
                    candidate_name(candidate): float(weight)
                    for candidate, weight in zip(candidates, weights) if weight > 1e-7
                },
                "weight_sum": float(weights.sum()),
                "pair_weight": float(sum(
                    weight for candidate, weight in zip(candidates, weights)
                    if candidate["type"] == "pair"
                )),
                "min_strict": float(min(exact.values())),
                "gain_2024": float(exact["2024/all"]), "metrics": exact,
            })
    solutions.sort(key=lambda item: (item["min_strict"], item["gain_2024"]), reverse=True)
    output = {
        "candidates": candidates, "constraint_count": len(curves),
        "baseline_solution": baseline, "solutions": solutions,
    }
    path = ROOT / "research/v25_r_combined_portfolio.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "constraint_count": len(curves), "candidates": candidates,
        "top": solutions[:15],
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
