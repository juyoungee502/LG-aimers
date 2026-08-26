"""Screen chronological sample-weight policies for the 2024 CatBoost fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)


# feature_engineering does not own this list in older revisions.
MODEL_CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="research/cat_weighting_2024.npz")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument(
        "--policies", default="all",
        help="Comma-separated policy names, or 'all'.",
    )
    return parser.parse_args()


def bss(y, prediction):
    rate = float(np.mean(y))
    return 100000.0 * (
        1.0 - np.mean((y - np.clip(prediction, .005, .995)) ** 2) / (rate * (1.0 - rate))
    )


def policies(seasons, reference_year):
    age = reference_year - seasons.astype(np.float64)
    result = {"uniform": np.ones(len(seasons), np.float32)}
    for half_life in (10., 5., 3., 2., 1.):
        result[f"half_life_{half_life:g}"] = np.exp(-np.log(2.) * age / half_life).astype(np.float32)
    for multiplier in (.0, .25, .5, 1.5):
        weight = np.ones(len(seasons), np.float32)
        weight[seasons == reference_year] = multiplier
        result[f"season_latest_x{multiplier:g}"] = weight
    return result


def main():
    args = arguments()
    raw = pd.read_csv(Path(args.data_dir) / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in MODEL_CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    season = raw["season"].to_numpy(np.int16)
    train_mask = season < args.validation_year
    valid_mask = season == args.validation_year
    x_train, x_valid = features.loc[train_mask], features.loc[valid_mask]
    y_train, y_valid = target[train_mask], target[valid_mask]
    settings = policies(season[train_mask], args.validation_year - 1)
    if args.policies != "all":
        selected = {value.strip() for value in args.policies.split(",") if value.strip()}
        unknown = selected.difference(settings)
        if unknown:
            raise ValueError(f"Unknown policies: {sorted(unknown)}")
        settings = {name: settings[name] for name in settings if name in selected}
    names, matrices, reports = [], [], {}
    for name, weight in settings.items():
        seed_predictions = []
        for seed in (42, 43, 44):
            params = dict(
                iterations=1200, learning_rate=.02, depth=6, loss_function="Logloss",
                eval_metric="Logloss", l2_leaf_reg=100., random_strength=1.,
                random_seed=seed, border_count=32, allow_writing_files=False,
                verbose=0, task_type=args.task_type, thread_count=args.threads,
            )
            if args.task_type == "GPU":
                params["devices"] = args.devices
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, sample_weight=weight)
            seed_predictions.append(model.predict_proba(x_valid)[:, 1])
        prediction = np.mean(seed_predictions, axis=0)
        names.append(name); matrices.append(prediction)
        correlation_score = 100000. * np.corrcoef(y_valid, prediction)[0, 1] ** 2
        reports[name] = {
            "bss": bss(y_valid, prediction),
            "affine_upper": correlation_score,
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
        }
        print(name, reports[name], flush=True)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, predictions=np.column_stack(matrices), target=y_valid,
        names=np.asarray(names), reports_json=np.asarray(json.dumps(reports)),
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
