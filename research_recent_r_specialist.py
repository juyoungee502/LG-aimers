"""Screen previous-season regular-league experts over v19."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed, depth, categorical=False):
    result = dict(
        iterations=1200, learning_rate=.02, depth=depth, l2_leaf_reg=100.,
        random_strength=1., border_count=32, loss_function="Logloss",
        eval_metric="Logloss", task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )
    if categorical:
        result.update(max_ctr_complexity=1, one_hot_max_size=32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    regular_all = data["game_type"].eq("R").to_numpy()
    valid = seasons == args.valid_year

    specs = [
        ("last1_r_d4", (seasons == args.valid_year - 1) & regular_all, 4, False),
        ("last1_r_d6", (seasons == args.valid_year - 1) & regular_all, 6, False),
        ("last2_r_d6", (seasons >= args.valid_year - 2) & regular_all, 6, False),
        ("last1_all_d6", seasons == args.valid_year - 1, 6, False),
        ("last1_r_cat_d6", (seasons == args.valid_year - 1) & regular_all, 6, True),
    ]
    predictions = []
    for offset, (name, train, depth, categorical) in enumerate(specs):
        train = train & (seasons < args.valid_year)
        model = CatBoostClassifier(**parameters(1200 + offset + args.valid_year, depth, categorical))
        fit_kwargs = {"cat_features": CAT_COLUMNS} if categorical else {}
        model.fit(features.loc[train], target[train], **fit_kwargs)
        prediction = model.predict_proba(features.loc[valid])[:, 1]
        predictions.append(prediction)
        print(f"Recent R model complete: {name}, rows={train.sum()}", flush=True)

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(float)
    base = oof["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for (name, *_), specialist in zip(specs, predictions):
        for weight in np.arange(0., .401, .01):
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
    output = root / f"research/recent_r_{args.valid_year}.npz"
    np.savez_compressed(
        output, names=np.asarray([spec[0] for spec in specs]),
        predictions=np.column_stack(predictions).astype(np.float32),
        target=y.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({"year": args.valid_year, "top": reports[:50]}, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
