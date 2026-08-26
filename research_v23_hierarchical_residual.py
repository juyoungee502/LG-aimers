"""Validate a hierarchical-prior residual CatBoost as an independent v23 axis.

This is a clean-room implementation of a standard empirical-Bayes residual
learner: establish a row-local pitcher prior from official as-of features, then
learn only ``target - prior`` with chronological sample weighting.  Every
validation feature uses target seasons strictly earlier than its row season.
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


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
VARIANTS = {
    "decay55": {"decay": .55, "iterations": 220},
    "decay30": {"decay": .30, "iterations": 220},
    "recent2": {"decay": .55, "iterations": 260, "window": 2},
}


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def hierarchical_prior(features: pd.DataFrame, raw: pd.DataFrame, level: float):
    recent = raw[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].apply(pd.to_numeric, errors="coerce")
    recent_std = recent.std(axis=1).fillna(.15).clip(0., .5).to_numpy(float)
    career_n = raw["asof_pitcher_n"].fillna(0.).clip(lower=0.).to_numpy(float)
    career_rate = raw["asof_pitcher_success_rate"].fillna(level).to_numpy(float)
    dynamic_strength = np.clip(
        55. + 220. * recent_std + 40. / (1. + np.log1p(career_n)), 50., 180.,
    )
    career = (
        career_rate * career_n + level * dynamic_strength
    ) / (career_n + dynamic_strength)

    season_n = features["pitcher_season_n"].fillna(0.).clip(lower=0.).to_numpy(float)
    season_rate = features["pitcher_season_success_rate"].fillna(level).to_numpy(float)
    season = (season_rate * season_n + level * 30.) / (season_n + 30.)
    reliability = season_n / (season_n + 80.)
    weight = .15 + .30 * reliability
    return np.clip(career + weight * (season - career), .01, .99)


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
    target = target_series.to_numpy()
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    level = float(target[train].mean())

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=level)
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = features.drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    prior = hierarchical_prior(features, raw, level)

    predictions, names = [], []
    for offset, (name, config) in enumerate(VARIANTS.items()):
        fit = train.copy()
        if config.get("window"):
            fit &= seasons >= args.valid_year - int(config["window"])
        age = (args.valid_year - 1) - seasons[fit].astype(float)
        sample_weight = np.power(float(config["decay"]), age).astype(np.float32)
        model = CatBoostRegressor(
            iterations=int(config["iterations"]), depth=8, learning_rate=.035,
            loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=12.,
            random_strength=.35, bootstrap_type="Bernoulli", subsample=.85,
            one_hot_max_size=16, border_count=32, task_type="GPU", devices="0",
            random_seed=8300 + args.valid_year * 10 + offset,
            allow_writing_files=False, verbose=0,
        )
        model.fit(
            features.loc[fit], target[fit] - prior[fit],
            sample_weight=sample_weight, cat_features=CAT_COLUMNS,
        )
        correction = model.predict(features.loc[valid])
        predictions.append(np.clip(prior[valid] + correction, .005, .995))
        names.append(name)
        print(
            f"Hierarchical residual complete: year={args.valid_year} "
            f"variant={name} rows={fit.sum()}", flush=True,
        )

    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(float)
    base = oof["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v23 OOF rows do not align")
    rows = raw.loc[valid].reset_index(drop=True)
    masks = segment_masks(rows)

    matrix = np.column_stack(predictions)
    candidates = {
        **{name: matrix[:, index] for index, name in enumerate(names)},
        "mean_55_30": matrix[:, :2].mean(axis=1),
        "mean_all": matrix.mean(axis=1),
    }
    reports = []
    for name, candidate in candidates.items():
        direction = logit(candidate) - logit(base)
        for weight in np.arange(-.10, .601, .01):
            prediction = sigmoid(logit(base) + weight * direction)
            gains = {
                label: bss(y[mask], prediction[mask]) - bss(y[mask], base[mask])
                for label, mask in masks.items() if mask.any()
            }
            reports.append({
                "name": name, "weight": float(weight), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_month": min(
                    gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                ),
                "standalone_bss": bss(y, candidate),
                "standalone_mean": float(candidate.mean()),
                "target_mean": float(y.mean()),
                "base_mean": float(base.mean()),
            })
    reports.sort(
        key=lambda row: (
            min(row["min_half"], row["min_month"]),
            row["gains"]["all"],
        ), reverse=True,
    )
    output = root / "research" / f"v23_hierarchical_residual_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        names=np.asarray(names), predictions=matrix.astype(np.float32),
        prior=prior[valid].astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({"year": args.valid_year, "top": reports[:50]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
