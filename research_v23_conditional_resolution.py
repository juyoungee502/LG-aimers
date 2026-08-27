"""Learn within-context resolution after removing seasonal regime levels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


SEEDS = (4501, 4502, 4503)
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
CONTEXTS = {
    "regime_count": ("game_type", "count_state"),
    "regime_count_hands": (
        "game_type", "count_state", "pitcher_hand", "batter_hand",
    ),
    "regime_count_runners": ("game_type", "count_state", "runner_gate"),
}


def context_frame(raw):
    result = pd.DataFrame(index=raw.index)
    result["season"] = raw["season"].to_numpy()
    result["game_type"] = raw["game_type"].astype(str).to_numpy()
    result["count_state"] = (
        raw["balls_before"].to_numpy(np.int16) * 3
        + raw["strikes_before"].to_numpy(np.int16)
    )
    result["pitcher_hand"] = raw["pitcher_hand"].to_numpy()
    result["batter_hand"] = raw["batter_hand"].to_numpy()
    result["runner_gate"] = raw["num_runners_on"].gt(0).to_numpy(np.int8)
    return result


def center_predictions(prediction, context, train, valid, keys):
    source = context.loc[train, list(keys)].copy()
    source["prediction"] = prediction[train]
    table = source.groupby(list(keys), observed=True)["prediction"].mean().reset_index()
    fallback = source.groupby("game_type", observed=True)["prediction"].mean().to_dict()
    query = context.loc[valid, list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    missing = query["prediction"].isna()
    query.loc[missing, "prediction"] = query.loc[missing, "game_type"].map(fallback)
    return query["prediction"].fillna(float(source["prediction"].mean())).to_numpy(float)


def segment_masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    context = context_frame(raw)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features).drop(columns=["season", "game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    directions = {}
    for mode_offset, (mode, keys) in enumerate(CONTEXTS.items()):
        label_frame = context[["season", *keys]].copy()
        label_frame["target"] = target
        center = label_frame.groupby(
            ["season", *keys], observed=True,
        )["target"].transform("mean").to_numpy(float)
        label = target - center
        valid_members = []
        source_members = []
        for seed in SEEDS:
            model = CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=.025,
                loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=150.,
                random_strength=1., border_count=32,
                bootstrap_type="Bayesian", bagging_temperature=.5,
                task_type="GPU", devices="0",
                random_seed=seed + 100 * mode_offset,
                allow_writing_files=False, verbose=0,
            )
            model.fit(features.loc[train], label[train])
            valid_members.append(model.predict(features.loc[valid]))
            source_members.append(model.predict(features.loc[train]))
            print(
                f"Conditional resolution complete: year={args.valid_year} "
                f"mode={mode} seed={seed}", flush=True,
            )
        valid_prediction = np.mean(valid_members, axis=0)
        all_prediction = np.zeros(len(raw), dtype=float)
        all_prediction[train] = np.mean(source_members, axis=0)
        all_prediction[valid] = valid_prediction
        frozen_center = center_predictions(all_prediction, context, train, valid, keys)
        directions[mode] = valid_prediction - frozen_center

    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(float)
    base = oof["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v23 OOF rows do not align")
    rows = raw.loc[valid].reset_index(drop=True)
    masks = segment_masks(rows)
    reports = []
    for mode, raw_direction in directions.items():
        for gate_name, active in (
            ("all", np.ones(len(y), dtype=bool)),
            ("regular", masks["regular"]), ("futures", masks["futures"]),
        ):
            direction = raw_direction * active
            for weight in np.arange(-.50, 1.501, .025):
                candidate = np.clip(base + weight * direction, .005, .995)
                gains = {
                    name: bss(y[mask], candidate[mask]) - bss(y[mask], base[mask])
                    for name, mask in masks.items() if mask.any()
                }
                reports.append({
                    "mode": mode, "gate": gate_name, "weight": float(weight),
                    "gains": gains,
                    "min_half": min(gains["first_half"], gains["second_half"]),
                    "min_month": min(
                        gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                    ),
                })
    reports.sort(
        key=lambda row: (
            min(row["min_half"], row["min_month"]), row["gains"]["all"],
        ), reverse=True,
    )
    output = root / f"research/v23_conditional_resolution_{args.valid_year}.npz"
    np.savez_compressed(
        output, names=np.asarray(list(directions)),
        directions=np.column_stack(list(directions.values())).astype(np.float32),
        target=y.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "valid_year": args.valid_year, "feature_count": features.shape[1],
        "top": reports[:60],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
