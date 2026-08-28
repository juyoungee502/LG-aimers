"""Evaluate the forward-frozen v41 career correction on the v38 ensemble."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import training_history_arrays
from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v41_career_detrend import grouped_rates


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"


def scores(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    regular = game_type == "R"
    result["R"] = float(bss(target[regular], prediction[regular]))
    result["F"] = float(bss(target[~regular], prediction[~regular]))
    return result


def main():
    columns = [
        "season", "game_type", "pitcher_id", "batter_id", TARGET,
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_batter_n", "asof_batter_success_rate",
    ]
    raw = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw[TARGET].astype(np.float32)
    features = raw.drop(columns=[TARGET])
    p_base_n, _, b_base_n, _ = training_history_arrays(features, target_series)
    global_rates = raw.groupby("season", observed=True)[TARGET].mean().to_dict()
    type_rates = raw.groupby(
        ["season", "game_type"], observed=True,
    )[TARGET].mean().to_dict()

    valid = raw["season"].eq(2024).to_numpy()
    rows = raw.loc[valid].reset_index(drop=True)
    corrections = {}
    for id_col in ("pitcher_id", "batter_id"):
        seasonal = raw.groupby(
            ["season", "game_type", id_col], observed=True, sort=False,
        )[TARGET].agg(n="size", success="sum").reset_index()
        old, adjusted = grouped_rates(
            seasonal, id_col, 2024, "global", None,
            global_rates, type_rates,
        )
        corrections[id_col] = np.nan_to_num(
            rows[id_col].map(adjusted).to_numpy(float)
            - rows[id_col].map(old).to_numpy(float),
            nan=0.,
        )

    current_p_n = np.maximum(
        0., rows["asof_pitcher_n"].fillna(0).to_numpy(float) - p_base_n[valid]
    )
    current_b_n = np.maximum(
        0., rows["asof_batter_n"].fillna(0).to_numpy(float) - b_base_n[valid]
    )
    strength = 200.
    direction = (
        .75 * strength / (current_p_n + strength) * corrections["pitcher_id"]
        + .25 * strength / (current_b_n + strength) * corrections["batter_id"]
    )

    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        target = archive["target"][active].astype(float)
        v24 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        v23 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
    with np.load(
        ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
    ) as archive:
        direct = archive["prediction"].astype(float)
    first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
    v38 = sigmoid(.90 * logit(first) + .10 * logit(direct))

    if not np.allclose(target, target_series.to_numpy(float)[valid]):
        raise ValueError("2024 raw rows do not align")
    blocks = masks(len(target))
    game_type = rows["game_type"].to_numpy(str)
    baselines = {
        "v23": scores(target, v23, blocks, game_type),
        "v38": scores(target, v38, blocks, game_type),
    }
    reports = []
    for weight in np.round(np.arange(-.10, .301, .025), 3):
        prediction = np.clip(v38 + weight * direction, .005, .995)
        result = scores(target, prediction, blocks, game_type)
        reports.append({
            "weight": float(weight),
            "scores": result,
            "gains_v38": {
                name: result[name] - baselines["v38"][name] for name in result
            },
            "gains_v23": {
                name: result[name] - baselines["v23"][name] for name in result
            },
        })
    frozen = next(row for row in reports if row["weight"] == .10)
    best_score = max(reports, key=lambda row: row["scores"]["all"])
    best_robust = max(
        reports,
        key=lambda row: (
            min(row["gains_v38"][f"q{i}"] for i in range(1, 5)),
            row["scores"]["all"],
        ),
    )
    report = {
        "v41_frozen_config": {
            "adjustment": "global", "half_life": None,
            "strength": 200., "mixture": "p75_b25", "weight": .10,
        },
        "direction": {
            "mean": float(direction.mean()),
            "std": float(direction.std()),
            "min": float(direction.min()),
            "max": float(direction.max()),
            "nonzero_fraction": float(np.mean(direction != 0.)),
        },
        "baselines": baselines,
        "frozen": frozen,
        "best_score": best_score,
        "best_robust": best_robust,
    }
    output = ROOT / "research/v42_v38_career_detrend.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
