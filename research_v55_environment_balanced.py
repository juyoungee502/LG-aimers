"""Test command models that are deliberately insensitive to yearly rosters.

The current pitch result is reconstructed only for historical training rows.
Raw player IDs, team IDs, and the team matchup are removed from every model.
The variants differ only in how much influence each historical environment
(season and game type) receives during fitting.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas.errors import PerformanceWarning

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss, reconstruct_labels
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent
CATEGORICAL_COLUMNS = ("base_state", "game_dayofweek")
MODES = ("half_life", "equal_season", "equal_environment")
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed, mode):
    return dict(
        iterations=args.iterations, learning_rate=.018, depth=7,
        l2_leaf_reg=420., random_strength=2.8, bagging_temperature=.7,
        border_count=32, bootstrap_type="Bayesian", loss_function="MultiClass",
        eval_metric="MultiClass", random_seed=5500 + seed + 1000 * MODES.index(mode),
        task_type=args.task_type, devices=args.devices,
        thread_count=args.threads, gpu_ram_part=.90,
        allow_writing_files=False, verbose=100,
    ) if args.task_type == "GPU" else dict(
        iterations=args.iterations, learning_rate=.018, depth=7,
        l2_leaf_reg=420., random_strength=2.8, bagging_temperature=.7,
        border_count=32, bootstrap_type="Bayesian", loss_function="MultiClass",
        eval_metric="MultiClass", random_seed=5500 + seed + 1000 * MODES.index(mode),
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )


def environment_weights(raw, train, valid_year, mode):
    season = raw["season"].to_numpy(np.int16)
    if mode == "half_life":
        age = (valid_year - 1) - season[train].astype(float)
        return np.exp(-np.log(2.) * age / 2.).astype(np.float32)

    columns = ["season"]
    if mode == "equal_environment":
        columns.append("game_type")
    groups = raw.loc[train, columns].astype(str)
    counts = groups.groupby(columns, observed=True)[columns[0]].transform("size")
    weight = 1. / counts.to_numpy(float)
    weight *= len(weight) / weight.sum()
    return weight.astype(np.float32)


def score(target, prediction, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in masks(len(target)).items()
    }
    for regime in ("R", "F"):
        active = game_type == regime
        result[regime] = float(bss(target[active], prediction[active]))
    return result


def main():
    args = arguments()
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    complete = labels[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    reverse = labels["reverse"].fillna(0).eq(1).to_numpy()
    middle = labels["middle"].fillna(0).eq(1).to_numpy()
    command = np.full(len(raw), -1, dtype=np.int8)
    command[complete & (target == 1)] = 0
    command[complete & (target == 0) & reverse & ~middle] = 1
    command[complete & (target == 0) & ~reverse & middle] = 2
    command[complete & (target == 0) & reverse & middle] = 3
    command[complete & (target == 0) & ~reverse & ~middle] = 4

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    features = features.drop(columns=[
        column for column in (
            "pitcher_id", "batter_id", "team_matchup",
            "pitcher_team_id", "batter_team_id",
        ) if column in features
    ])
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    season = raw["season"].to_numpy(np.int16)
    train = (season < args.valid_year) & complete
    valid = season == args.valid_year
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    predictions = {}
    weight_diagnostics = {}
    for mode in args.modes:
        weights = environment_weights(raw, train, args.valid_year, mode)
        members = []
        for seed_index in range(args.n_seeds):
            print(json.dumps({
                "mode": mode, "seed": seed_index + 1,
                "train_rows": int(train.sum()), "features": int(features.shape[1]),
            }), flush=True)
            model = CatBoostClassifier(**parameters(args, 101 * seed_index, mode))
            model.fit(
                features.loc[train], command[train], sample_weight=weights,
                cat_features=list(CATEGORICAL_COLUMNS),
            )
            probability = model.predict_proba(features.loc[valid])
            success_column = int(np.flatnonzero(np.asarray(model.classes_, int) == 0)[0])
            members.append(probability[:, success_column])
        predictions[mode] = np.mean(members, axis=0)
        frame = pd.DataFrame({
            "season": raw.loc[train, "season"].to_numpy(),
            "game_type": raw.loc[train, "game_type"].astype(str).to_numpy(),
            "weight": weights,
        })
        by_environment = frame.groupby(
            ["season", "game_type"], observed=True,
        )["weight"].sum()
        weight_diagnostics[mode] = {
            "by_season": {
                str(key): float(value)
                for key, value in frame.groupby("season")["weight"].sum().items()
            },
            "by_environment": {
                f"{season_key}:{regime}": float(value)
                for (season_key, regime), value in by_environment.items()
            },
        }

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v38 rows do not align")
    baseline = score(fold_target, base, game_type)
    reports = []
    for name, prediction in predictions.items():
        direction = logit(np.clip(prediction, .005, .995)) - logit(base)
        for gate in ("all", "R", "F"):
            selected = np.ones(len(base), bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(-.05, .201, .0125), 4):
                candidate = base.copy()
                candidate[selected] = sigmoid(
                    logit(base[selected]) + weight * direction[selected]
                )
                scores = score(fold_target, candidate, game_type)
                gains = {key: scores[key] - baseline[key] for key in scores}
                reports.append({
                    "name": name, "gate": gate, "weight": float(weight),
                    "scores": scores, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })

    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["R"], row["gains"]["F"]),
        row["gains"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "n_seeds": args.n_seeds,
        "raw_player_ids_used": False, "raw_team_ids_used": False,
        "baseline": baseline, "weight_diagnostics": weight_diagnostics,
        "standalone": {
            name: score(fold_target, value, game_type)
            for name, value in predictions.items()
        },
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:30],
        "best_score": sorted(reports, key=lambda row: row["gains"]["all"], reverse=True)[:30],
    }
    output = ROOT / "research" / f"v55_environment_balanced_s{args.n_seeds}_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
