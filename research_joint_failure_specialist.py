"""Compare direct and joint-category failure models over the v19 ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss, reconstruct_labels


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed, loss):
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function=loss, eval_metric=loss,
        task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    usable = labels[["reverse", "middle", "wayoff"]].notna().all(axis=1).to_numpy()
    bits = (
        labels["reverse"].fillna(0).to_numpy(np.int8)
        + 2 * labels["middle"].fillna(0).to_numpy(np.int8)
        + 4 * labels["wayoff"].fillna(0).to_numpy(np.int8)
    )
    predictions = {}
    binary = CatBoostClassifier(**parameters(880 + args.valid_year, "Logloss"))
    binary.fit(features.loc[train], target[train])
    predictions["direct_binary"] = binary.predict_proba(features.loc[valid])[:, 1]

    joint_train = train & usable
    joint = CatBoostClassifier(**parameters(980 + args.valid_year, "MultiClass"))
    joint.fit(features.loc[joint_train], bits[joint_train])
    class_zero = list(joint.classes_).index(0)
    predictions["joint_multiclass"] = joint.predict_proba(features.loc[valid])[:, class_zero]

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for name, specialist in predictions.items():
        for weight in np.arange(0., .301, .01):
            blended = base.copy()
            blended[regular] = sigmoid(
                (1. - weight) * logit(base[regular])
                + weight * logit(specialist[regular])
            )
            report = {
                "name": name, "weight": float(weight),
                "gain": bss(y, blended) - bss(y, base),
                "gain_first_half": bss(y[:midpoint], blended[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                "gain_second_half": bss(y[midpoint:], blended[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                "standalone_r_bss": bss(y[regular], specialist[regular]),
            }
            report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
            reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / f"research/joint_failure_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        names=np.asarray(list(predictions)), predictions=np.column_stack(list(predictions.values())).astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({"year": args.valid_year, "top": reports[:40]}, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
