"""Learn season-invariant within-regime resolution and add it to v23.

The regressor never learns annual target levels: labels are centered inside
each historical season (optionally season/game_type).  Its frozen train-side
prediction mean is removed at deployment, so it contributes row-level
resolution without guessing the private season's target rate.
"""
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


SEEDS = (4401, 4402, 4403)
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


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


def centered_target(raw, target, mode):
    frame = pd.DataFrame({
        "season": raw["season"].to_numpy(),
        "game_type": raw["game_type"].astype(str).to_numpy(),
        "target": target,
    })
    keys = ["season"] if mode == "season" else ["season", "game_type"]
    means = frame.groupby(keys, observed=True)["target"].transform("mean")
    return target - means.to_numpy(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = features.drop(columns=["season", "game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    game_type = raw["game_type"].astype(str).to_numpy()
    directions = {}
    for mode_offset, mode in enumerate(("season", "season_game_type")):
        label = centered_target(raw, target, mode)
        members = []
        source_predictions = []
        for seed_offset, seed in enumerate(SEEDS):
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
            members.append(model.predict(features.loc[valid]))
            source_predictions.append(model.predict(features.loc[train]))
            print(
                f"Centered resolution complete: year={args.valid_year} "
                f"mode={mode} seed={seed}", flush=True,
            )
        prediction = np.mean(members, axis=0)
        source_prediction = np.mean(source_predictions, axis=0)
        direction = prediction.copy()
        for regime in ("R", "F"):
            source_mask = train & (game_type == regime)
            valid_mask = game_type[valid] == regime
            direction[valid_mask] -= float(source_prediction[
                game_type[train] == regime
            ].mean())
        directions[mode] = direction

    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(float)
    base = oof["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v23 OOF rows do not align")
    rows = raw.loc[valid].reset_index(drop=True)
    masks = segment_masks(rows)
    regular = masks["regular"]
    reports = []
    for mode, raw_direction in directions.items():
        for gate_name in ("all", "regular"):
            direction = raw_direction.copy()
            if gate_name == "regular":
                direction[~regular] = 0.0
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
                    "direction_mean": float(direction.mean()),
                    "direction_std": float(direction.std()),
                })
    reports.sort(
        key=lambda row: (
            min(row["min_half"], row["min_month"]), row["gains"]["all"],
        ), reverse=True,
    )
    output = root / f"research/v23_centered_resolution_{args.valid_year}.npz"
    np.savez_compressed(
        output, names=np.asarray(list(directions)),
        directions=np.column_stack(list(directions.values())).astype(np.float32),
        target=y.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "valid_year": args.valid_year, "feature_count": features.shape[1],
        "top": reports[:50],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
