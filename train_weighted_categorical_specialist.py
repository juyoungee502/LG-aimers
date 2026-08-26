"""Train recency-weighted native-categorical count specialists for v15."""
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


CAT_COLUMNS = [
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
        random_seed=seed, border_count=32, max_ctr_complexity=1,
        one_hot_max_size=32, allow_writing_files=False, verbose=0,
        task_type=args.task_type, thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def routed_fold(features, target, seasons, valid_year, args):
    train, valid = seasons < valid_year, seasons == valid_year
    train_gate = features.loc[train, "two_strike"].to_numpy().astype(bool)
    valid_gate = features.loc[valid, "two_strike"].to_numpy().astype(bool)
    seed_predictions = []
    for index, seed in enumerate((142, 143, 144)):
        prediction = np.empty(valid.sum(), dtype=np.float64)
        for label, gate_value in (("other", False), ("two_strike", True)):
            train_rows = train.copy()
            train_rows[train] = train_gate == gate_value
            valid_rows = valid_gate == gate_value
            model = CatBoostClassifier(**parameters(args, seed + 10 * int(gate_value)))
            model.fit(
                features.loc[train_rows], target[train_rows],
                sample_weight=weights(seasons[train_rows], valid_year - 1),
                cat_features=CAT_COLUMNS,
            )
            prediction[valid_rows] = model.predict_proba(
                features.loc[valid].loc[valid_rows]
            )[:, 1]
        seed_predictions.append(prediction)
    return np.mean(seed_predictions, axis=0)


def main():
    args = arguments()
    raw = pd.read_csv(Path(args.data_dir) / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)

    predictions, targets, years = [], [], []
    for valid_year in (2023, 2024):
        valid = seasons == valid_year
        prediction = routed_fold(features, target, seasons, valid_year, args)
        predictions.append(prediction)
        targets.append(target[valid])
        years.append(seasons[valid])
        print(
            f"Weighted categorical specialist fold {valid_year}: "
            f"BSS={bss(target[valid], prediction):.4f}", flush=True,
        )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_reference = int(seasons.max())
    gate = features["two_strike"].to_numpy().astype(bool)
    for index, seed in enumerate((142, 143, 144)):
        for label, gate_value in (("other", False), ("two_strike", True)):
            mask = gate == gate_value
            model = CatBoostClassifier(**parameters(args, seed + 10 * int(gate_value)))
            model.fit(
                features.loc[mask], target[mask],
                sample_weight=weights(seasons[mask], final_reference),
                cat_features=CAT_COLUMNS,
            )
            model.save_model(
                str(model_dir / f"catboost_weighted_categorical_{label}_{index}.cbm")
            )

    diagnostic_dir = Path(args.diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    output = diagnostic_dir / "v15_weighted_categorical_oof_predictions.npz"
    np.savez_compressed(
        output, prediction=np.concatenate(predictions),
        target=np.concatenate(targets), season=np.concatenate(years),
    )
    print(f"Saved weighted categorical specialists and OOF to {output}")


if __name__ == "__main__":
    main()
