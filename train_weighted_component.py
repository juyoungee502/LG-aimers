"""Train the half-life-3 CatBoost component for v14 OOF and final inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)


MODEL_CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="submit/model")
    parser.add_argument("--diagnostic-dir", default="outputs")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def weights(seasons, reference):
    age = np.maximum(0., reference - seasons.astype(np.float64))
    return np.exp(-np.log(2.) * age / 3.).astype(np.float32)


def parameters(args, seed):
    result = dict(
        iterations=1200, learning_rate=.02, depth=6, loss_function="Logloss",
        eval_metric="Logloss", l2_leaf_reg=100., random_strength=1.,
        random_seed=seed, border_count=32, allow_writing_files=False,
        verbose=0, task_type=args.task_type, thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
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

    predictions, targets, years = [], [], []
    for valid_year in (2023, 2024):
        train_mask, valid_mask = season < valid_year, season == valid_year
        seed_predictions = []
        for seed in (42, 43, 44):
            model = CatBoostClassifier(**parameters(args, seed))
            model.fit(
                features.loc[train_mask], target[train_mask],
                sample_weight=weights(season[train_mask], valid_year - 1),
            )
            seed_predictions.append(model.predict_proba(features.loc[valid_mask])[:, 1])
        prediction = np.mean(seed_predictions, axis=0)
        predictions.append(prediction); targets.append(target[valid_mask]); years.append(season[valid_mask])
        rate = float(target[valid_mask].mean())
        score = 100000. * (
            1. - np.mean((target[valid_mask] - prediction) ** 2) / (rate * (1. - rate))
        )
        print(f"Weighted CatBoost fold {valid_year}: BSS={score:.4f}", flush=True)

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    reference = int(season.max())
    final_weight = weights(season, reference)
    for index, seed in enumerate((42, 43, 44)):
        model = CatBoostClassifier(**parameters(args, seed))
        model.fit(features, target, sample_weight=final_weight)
        model.save_model(str(model_dir / f"catboost_weighted_{index}.cbm"))

    diagnostic_dir = Path(args.diagnostic_dir); diagnostic_dir.mkdir(parents=True, exist_ok=True)
    output = diagnostic_dir / "v14_weighted_oof_predictions.npz"
    np.savez_compressed(
        output, prediction=np.concatenate(predictions),
        target=np.concatenate(targets), season=np.concatenate(years),
    )
    print(f"Saved weighted models to {model_dir} and OOF to {output}")


if __name__ == "__main__":
    main()
