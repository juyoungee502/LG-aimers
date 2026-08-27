"""Find a conservative mixture of the independently stable v25 residual effects.

The search is deliberately restricted to four effects which were positive on
all four forward transfers.  It never reads test data or leaderboard scores.
Weights are ranked by the worst chronological/season segment, rather than the
full-year mean, to reduce validation-period overfitting.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import TARGET_COL
from research_inferred_pitch_priors import bss
from research_v25_transfer_portfolio import direction, feature_frames


ROOT = Path(__file__).resolve().parent
FEATURES = (
    "asof_pitcher_success_rate",
    "pitcher_middle_season_rate",
    "pitcher_success_x_runners",
    "asof_pitcher_prev5_game_success_rate",
)
WEIGHT_GRID = (
    (0.50, 0.75, 1.00, 1.25),
    (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
)


def gain(target, base, correction):
    return bss(target, np.clip(base + correction, .005, .995)) - bss(target, base)


def masks_for(rows: pd.DataFrame, *, prefix: str):
    n = len(rows)
    position = np.arange(n)
    masks = {
        f"{prefix}/all": np.ones(n, dtype=bool),
        f"{prefix}/half_1": position < n // 2,
        f"{prefix}/half_2": position >= n // 2,
        f"{prefix}/q1": position < n // 4,
        f"{prefix}/q2": (position >= n // 4) & (position < n // 2),
        f"{prefix}/q3": (position >= n // 2) & (position < 3 * n // 4),
        f"{prefix}/q4": position >= 3 * n // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        # Very small postseason tails are not used to select a regular-season
        # effect.  All ordinary monthly groups are retained.
        if active.sum() >= 100:
            masks[f"{prefix}/month_{int(month)}"] = active
    return masks


def metrics(target, base, correction, masks):
    return {
        name: gain(target[active], base[active], correction[active])
        for name, active in masks.items()
    }


def load_specs():
    payload = json.loads(
        (ROOT / "research/v24_exhaustive_transfer.json").read_text(encoding="utf-8")
    )
    specs = {}
    for item in payload["top"]:
        if item["column"] in FEATURES and item["column"] not in specs:
            specs[item["column"]] = item
    missing = set(FEATURES) - set(specs)
    if missing:
        raise ValueError(f"Missing screened specifications: {sorted(missing)}")
    return [specs[column] for column in FEATURES]


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    numeric_all, categorical_all = feature_frames(raw, target_series)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}

    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([np.flatnonzero(seasons == year) for year in (2023, 2024)])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    numeric = numeric_all.iloc[positions].reset_index(drop=True)
    categorical = categorical_all.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    target = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    regular = rows["game_type"].eq("R").to_numpy()
    regular_indices = {value: np.flatnonzero(regular & (year == value)) for value in (2023, 2024)}
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in regular_indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", regular_indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", regular_indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
    )
    specs = load_specs()

    transfer_data = {}
    for label, source, valid in transfers:
        transfer_data[label] = {
            "target": target[valid],
            "base": base[valid],
            "directions": np.stack([
                direction(spec, source, valid, numeric, categorical, target, base)
                for spec in specs
            ]),
            "masks": masks_for(rows.iloc[valid].reset_index(drop=True), prefix=label),
        }

    # The full 2023 -> 2024 replay supplies the month/quarter stability gate.
    source, valid = regular_indices[2023], regular_indices[2024]
    full = {
        "target": target[valid], "base": base[valid],
        "directions": np.stack([
            direction(spec, source, valid, numeric, categorical, target, base)
            for spec in specs
        ]),
        "masks": masks_for(rows.iloc[valid].reset_index(drop=True), prefix="2024"),
    }

    accepted = []
    evaluated = 0
    for weights in itertools.product(*WEIGHT_GRID):
        evaluated += 1
        values = {}
        for label, item in transfer_data.items():
            correction = np.tensordot(weights, item["directions"], axes=1)
            values.update(metrics(item["target"], item["base"], correction, item["masks"]))
        correction = np.tensordot(weights, full["directions"], axes=1)
        values.update(metrics(full["target"], full["base"], correction, full["masks"]))

        transfer_all = [values[f"{label}/all"] for label, _source, _valid in transfers]
        stability = [value for name, value in values.items() if name.startswith("2024/")]
        # Hard gate: every forward transfer, 2024 half/quarter and normal month
        # must improve.  2023 target subsegments are kept as an audit because
        # their source half is intentionally much smaller.
        if min(transfer_all) <= 0 or min(stability) <= 0:
            continue
        robust_values = transfer_all + stability
        # Prefer a strong worst segment, then a strong conservative 20th
        # percentile.  Mean gain is only the final tie breaker.
        rank = (
            float(min(robust_values)),
            float(np.quantile(robust_values, .20)),
            float(np.mean(transfer_all)),
            float(values["2024/all"]),
        )
        accepted.append({
            "rank": rank,
            "weights": {feature: float(weight) for feature, weight in zip(FEATURES, weights)},
            "metrics": values,
        })

    accepted.sort(key=lambda item: item["rank"], reverse=True)
    result = {
        "evaluated": evaluated,
        "accepted": len(accepted),
        "features": specs,
        "top": accepted[:30],
    }
    output = ROOT / "research/v25_robust_mix.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
