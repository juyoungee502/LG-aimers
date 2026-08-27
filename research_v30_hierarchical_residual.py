"""Independent hierarchical-residual channels with a Futures expert.

The model is intentionally compact and strongly regularized.  It predicts the
residual around a row-local empirical-Bayes pitcher baseline, using either
recency-weighted league history or Futures-only history.  The screen is fully
chronological and is designed as a diverse ensemble axis for v23, not as a
replacement selected on one validation season.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
CAT_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def hierarchical_base(raw, features, prior):
    recent = raw[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].apply(pd.to_numeric, errors="coerce")
    recent_std = recent.std(axis=1).fillna(.15).clip(0., .5).to_numpy(float)
    career_n = pd.to_numeric(
        raw["asof_pitcher_n"], errors="coerce",
    ).fillna(0.).clip(lower=0.).to_numpy(float)
    career_rate = pd.to_numeric(
        raw["asof_pitcher_success_rate"], errors="coerce",
    ).fillna(prior).to_numpy(float)
    strength = np.clip(
        55. + 220. * recent_std + 40. / (1. + np.log1p(career_n)),
        50., 180.,
    )
    career = (career_rate * career_n + prior * strength) / (
        career_n + strength
    )
    season_n = features["pitcher_season_n"].to_numpy(float)
    season_rate = features["pitcher_season_success_rate"].to_numpy(float)
    season = (season_rate * season_n + prior * 30.) / (season_n + 30.)
    reliability = season_n / (season_n + 80.)
    base = career + (.15 + .30 * reliability) * (season - career)
    return np.clip(base, .05, .95).astype(np.float32)


def parameters(args, seed, iterations):
    result = dict(
        iterations=iterations, depth=8, learning_rate=.035,
        loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=20.,
        random_strength=.35, border_count=32, bootstrap_type="Bernoulli",
        subsample=.85, one_hot_max_size=16, task_type=args.task_type,
        random_seed=seed, allow_writing_files=False, verbose=100,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def fit_predict(features, residual, train, valid_rows, weights, args, seed,
                iterations):
    model = CatBoostRegressor(**parameters(args, seed, iterations))
    model.fit(
        features.iloc[np.flatnonzero(train)], residual[train],
        sample_weight=weights, cat_features=list(CAT_COLUMNS),
    )
    return model.predict(features.iloc[valid_rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True,
                        choices=(2022, 2023, 2024))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig",
                      low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    seasons = raw["season"].to_numpy(np.int16)
    prior = float(target[seasons < args.valid_year].mean())
    history = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *history, global_prior=prior)
    add_training_component_features(features, raw)
    features = add_state_interactions(features).copy()
    base = hierarchical_base(raw, features, prior)
    features["_hierarchical_base"] = base
    features["_hierarchical_logit"] = np.log(base / (1. - base))
    features["_recent_weighted"] = (
        .55 * raw["asof_pitcher_prev1_game_success_rate"].fillna(prior)
        + .30 * raw["asof_pitcher_prev3_game_success_rate"].fillna(prior)
        + .15 * raw["asof_pitcher_prev5_game_success_rate"].fillna(prior)
    ).astype(np.float32)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    valid = seasons == args.valid_year
    valid_rows = np.flatnonzero(valid)
    regular = raw["game_type"].eq("R").to_numpy()
    futures = ~regular
    residual = (target - base).astype(np.float32)
    age = (args.valid_year - 1) - seasons.astype(float)
    specs = (
        ("all_decay55", seasons < args.valid_year, np.power(.55, age), 180),
        ("all_decay30", seasons < args.valid_year, np.power(.30, age), 200),
        ("recent_all", seasons == args.valid_year - 1,
         np.ones(len(raw)), 220),
        ("f_decay55", (seasons < args.valid_year) & futures,
         np.power(.55, age), 180),
        ("f_decay30", (seasons < args.valid_year) & futures,
         np.power(.30, age), 200),
        ("f_recent", (seasons == args.valid_year - 1) & futures,
         np.ones(len(raw)), 220),
    )
    corrections = []
    for offset, (name, train, sample_weight, iterations) in enumerate(specs):
        correction = fit_predict(
            features, residual, train, valid_rows,
            sample_weight[train].astype(np.float32), args,
            9001 + args.valid_year * 10 + offset, iterations,
        )
        corrections.append(correction)
        prediction = np.clip(base[valid] + correction, .005, .995)
        print(
            f"v30 {args.valid_year} {name}: train={train.sum()} "
            f"bss={bss(target[valid], prediction):.6f}", flush=True,
        )

    corrections = np.column_stack(corrections)
    names = np.asarray([spec[0] for spec in specs])
    full = base[valid] + .45 * corrections[:, 0] + .55 * corrections[:, 1]
    recent = base[valid] + corrections[:, 2]
    f_ensemble = (
        base[valid] + .35 * corrections[:, 3]
        + .40 * corrections[:, 4] + .25 * corrections[:, 5]
    )
    candidate = .75 * full + .25 * recent
    candidate[futures[valid]] = f_ensemble[futures[valid]]
    prediction_names = np.asarray(["full", "recent", "f_ensemble", "candidate"])
    predictions = np.column_stack([full, recent, f_ensemble, candidate])
    predictions = np.clip(predictions, .005, .995)
    for index, name in enumerate(prediction_names):
        print(f"v30 combined {args.valid_year} {name}: "
              f"bss={bss(target[valid], predictions[:, index]):.6f}",
              flush=True)

    output = ROOT / f"research/v30_hierarchical_residual_{args.valid_year}.npz"
    np.savez_compressed(
        output, correction_names=names,
        corrections=corrections.astype(np.float32),
        prediction_names=prediction_names,
        predictions=predictions.astype(np.float32),
        base=base[valid], target=target[valid].astype(np.float32),
        regular=regular[valid], season=seasons[valid],
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
