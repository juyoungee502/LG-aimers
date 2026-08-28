"""Complexity and checkpoint audit for the post-break F multiclass model."""
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
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="4,5,6")
    parser.add_argument("--checkpoints", default="400,700,1000")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, depth, seed, iterations):
    result = dict(
        iterations=iterations, learning_rate=.025, depth=depth,
        l2_leaf_reg=160., random_strength=2., bagging_temperature=.8,
        border_count=32, bootstrap_type="Bayesian", loss_function="MultiClass",
        eval_metric="MultiClass", random_seed=4900 + 101 * seed + 11 * depth,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def score(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    regular = game_type == "R"
    result["R"] = float(bss(target[regular], prediction[regular]))
    result["F"] = float(bss(target[~regular], prediction[~regular]))
    return result


def main():
    args = arguments()
    depths = tuple(int(value) for value in args.depths.split(","))
    checkpoints = tuple(sorted(int(value) for value in args.checkpoints.split(",")))
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    complete = labels[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    reverse = labels["reverse"].fillna(0).eq(1).to_numpy()
    middle = labels["middle"].fillna(0).eq(1).to_numpy()
    outcome = np.full(len(raw), -1, dtype=np.int8)
    outcome[complete & (target == 1)] = 0
    outcome[complete & (target == 0) & reverse & ~middle] = 1
    outcome[complete & (target == 0) & ~reverse & middle] = 2
    outcome[complete & (target == 0) & reverse & middle] = 3
    outcome[complete & (target == 0) & ~reverse & ~middle] = 4

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])
    for column in LOW_CARD_CATEGORIES:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    f_rows = raw["game_type"].eq("F").to_numpy()
    train = (seasons == 2023) & f_rows & complete
    valid = seasons == 2024
    valid_f = f_rows[valid]
    print(json.dumps({
        "depths": depths, "checkpoints": checkpoints,
        "n_seeds": args.n_seeds, "train_rows": int(train.sum()),
        "valid_F_rows": int(valid_f.sum()), "features": int(features.shape[1]),
    }), flush=True)

    all_predictions = {
        f"d{depth}_i{checkpoint}": []
        for depth in depths for checkpoint in checkpoints
    }
    for depth in depths:
        for seed in range(args.n_seeds):
            model = CatBoostClassifier(**parameters(
                args, depth, seed, max(checkpoints),
            ))
            model.fit(
                features.loc[train], outcome[train],
                cat_features=list(LOW_CARD_CATEGORIES),
            )
            class_order = np.asarray(model.classes_, dtype=int)
            success_position = int(np.flatnonzero(class_order == 0)[0])
            for checkpoint in checkpoints:
                probability = model.predict_proba(
                    features.loc[valid], ntree_end=checkpoint,
                )[:, success_position]
                all_predictions[f"d{depth}_i{checkpoint}"].append(probability)
            print(
                f"completed depth={depth}, seed={seed + 1}/{args.n_seeds}",
                flush=True,
            )
    predictions = {
        name: np.mean(values, axis=0)
        for name, values in all_predictions.items()
    }

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v38 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    reports = []
    for name, prediction in predictions.items():
        direction = logit(np.clip(prediction, .005, .995)) - logit(base)
        for weight in np.round(np.arange(0., .251, .025), 3):
            candidate = base.copy()
            candidate[valid_f] = sigmoid(
                logit(base[valid_f]) + weight * direction[valid_f]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {key: result[key] - baseline[key] for key in result}
            reports.append({
                "name": name, "weight": float(weight), "scores": result,
                "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["F"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "baseline": baseline,
        "standalone_F": {
            name: {
                "bss": float(bss(fold_target[valid_f], value[valid_f])),
                "mean": float(value[valid_f].mean()),
                "correlation_base": float(np.corrcoef(
                    value[valid_f], base[valid_f]
                )[0, 1]),
            } for name, value in predictions.items()
        },
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:60],
        "best_score": sorted(
            reports, key=lambda row: row["scores"]["all"], reverse=True,
        )[:60],
    }
    output = ROOT / "research/v49_regime_multiclass_complexity_2024.npz"
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
