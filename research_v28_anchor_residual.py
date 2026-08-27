"""GPU screen for season-adaptive, row-local residual models.

The target is decomposed into an anchor made only from official as-of recent
rates plus a low-capacity CatBoost residual. Anonymous player/team IDs are
excluded so that the learned count and state response can transfer by season.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from catboost import CatBoostRegressor

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)


ROOT = Path(__file__).resolve().parent
DROP_COLUMNS = {
    "season", "pitcher_id", "batter_id", "pitcher_team_id",
    "batter_team_id", "team_matchup",
}
warnings.filterwarnings("ignore", category=PerformanceWarning)


def parameters(seed, depth, args):
    result = dict(
        iterations=1000 if args.full else 120,
        learning_rate=.02 if args.full else .05,
        depth=depth,
        l2_leaf_reg=300.,
        random_strength=1.5,
        border_count=32,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type=args.task_type,
        random_seed=seed,
        allow_writing_files=False,
        verbose=100,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def recent_anchor(raw, features):
    values = []
    for name in (
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ):
        values.append(pd.to_numeric(raw[name], errors="coerce").to_numpy(float))
    prior = features["pitcher_prior_success_rate"].to_numpy(float)
    for index in range(len(values)):
        values[index] = np.where(np.isfinite(values[index]), values[index], prior)
    recent = .15 * values[0] + .35 * values[1] + .50 * values[2]
    season = features["pitcher_season_success_s100"].to_numpy(float)
    # Recent game form tracks the moving season level; the smoothed season
    # component reduces noise for pitchers with few current-season pitches.
    anchor = .70 * recent + .30 * season
    return np.clip(anchor, .05, .95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True,
                        choices=(2021, 2022, 2023, 2024))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    history = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *history, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    anchor = recent_anchor(raw, features)
    features["_recent_anchor"] = anchor.astype(np.float32)
    for column in DROP_COLUMNS:
        if column in features:
            features.drop(columns=column, inplace=True)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    seasons = raw["season"].to_numpy(np.int16)
    regular = raw["game_type"].eq("R").to_numpy()
    valid = seasons == args.valid_year
    prediction_rows = np.flatnonzero(valid)
    residual_target = (target - anchor).astype(np.float32)
    specs = (
        ("last2_r_d4", (seasons >= args.valid_year - 2) & regular, 4),
        ("last2_r_d6", (seasons >= args.valid_year - 2) & regular, 6),
        ("all_r_d5", regular.copy(), 5),
    )
    predictions = []
    for offset, (name, train, depth) in enumerate(specs):
        train &= seasons < args.valid_year
        model = CatBoostRegressor(**parameters(2800 + args.valid_year * 10 + offset,
                                                depth, args))
        if name.startswith("all_"):
            age = (args.valid_year - 1) - seasons[train].astype(float)
            sample_weight = np.exp(-np.log(2.) * age / 3.).astype(np.float32)
        else:
            sample_weight = None
        model.fit(
            features.iloc[np.flatnonzero(train)], residual_target[train],
            sample_weight=sample_weight,
        )
        correction = model.predict(features.iloc[prediction_rows])
        prediction = np.clip(anchor[valid] + correction, .005, .995)
        predictions.append(prediction)
        print(
            f"anchor residual complete: {name}, year={args.valid_year}, "
            f"train_rows={train.sum()}", flush=True,
        )

    output = ROOT / f"research/v28_anchor_residual_{args.valid_year}.npz"
    np.savez_compressed(
        output,
        names=np.asarray([x[0] for x in specs]),
        predictions=np.column_stack(predictions).astype(np.float32),
        anchor=anchor[valid].astype(np.float32),
        target=target[valid].astype(np.float32),
        regular=regular[valid],
        season=np.full(valid.sum(), args.valid_year, dtype=np.int16),
        feature_names=np.asarray(features.columns),
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
