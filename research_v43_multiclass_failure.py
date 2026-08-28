"""Train a coherent five-class command-outcome model.

The next within-season cumulative state recovers labels for historical train
rows only.  Reverse and middle can overlap, so failures are partitioned into
reverse-only, middle-only, both, and other.  Together with success these five
classes avoid the probability-sum inconsistency of three independent binary
failure models.  Evaluation remains chronological.
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
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
CLASS_NAMES = (
    "success", "reverse_only", "middle_only", "reverse_middle", "wayoff",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1300)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed):
    result = dict(
        iterations=args.iterations,
        learning_rate=.018,
        depth=args.depth,
        l2_leaf_reg=400.,
        random_strength=2.8,
        bagging_temperature=.7,
        border_count=32,
        bootstrap_type="Bayesian",
        loss_function="MultiClass",
        eval_metric="MultiClass",
        task_type=args.task_type,
        thread_count=args.threads,
        random_seed=4300 + 101 * seed,
        allow_writing_files=False,
        verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, gpu_ram_part=.90)
    return result


def score_blocks(target, prediction, blocks, game_type):
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
    train = (seasons < args.valid_year) & (outcome >= 0)
    valid = seasons == args.valid_year
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    print(json.dumps({
        "valid_year": args.valid_year,
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "feature_count": int(features.shape[1]),
        "train_class_counts": {
            CLASS_NAMES[index]: int(np.sum(outcome[train] == index))
            for index in range(len(CLASS_NAMES))
        },
        "recovered_fraction": float(complete.mean()),
    }), flush=True)

    seed_predictions = []
    for seed in range(args.n_seeds):
        print(f"Training v43 multiclass seed {seed + 1}/{args.n_seeds}", flush=True)
        model = CatBoostClassifier(**parameters(args, seed))
        model.fit(
            features.loc[train], outcome[train],
            sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        probabilities = model.predict_proba(features.loc[valid])
        class_order = np.asarray(model.classes_, dtype=int)
        success_index = int(np.flatnonzero(class_order == 0)[0])
        seed_predictions.append(probabilities[:, success_index])
    prediction = np.mean(seed_predictions, axis=0)

    bases_oof = {}
    for version in (23, 24):
        with np.load(ROOT / f"outputs/v{version}_oof_predictions.npz") as archive:
            active = archive["season"] == args.valid_year
            if not np.allclose(archive["target"][active], target[valid]):
                raise ValueError(f"v{version} rows do not align")
            bases_oof[f"v{version}"] = np.clip(
                archive["blended"][active].astype(float), .005, .995,
            )
    fold_target = target[valid].astype(float)
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(fold_target))
    baselines = {
        name: score_blocks(fold_target, value, blocks, game_type)
        for name, value in bases_oof.items()
    }
    reports = []
    for base_name, base in bases_oof.items():
        direction = logit(prediction) - logit(base)
        for gate in ("all", "R", "F"):
            selected = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(-.10, .401, .025), 4):
                candidate = base.copy()
                candidate[selected] = sigmoid(
                    logit(base[selected]) + weight * direction[selected]
                )
                result = score_blocks(fold_target, candidate, blocks, game_type)
                gains = {
                    name: result[name] - baselines[base_name][name]
                    for name in result
                }
                reports.append({
                    "base": base_name, "gate": gate, "weight": float(weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })
    ranking = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"], row["gains"]["R"],
                row["gains"]["F"]),
            row["gains"]["all"],
        ), reverse=True,
    )
    score_ranking = sorted(
        reports, key=lambda row: row["scores"]["all"], reverse=True,
    )
    diagnostics = {
        "valid_year": args.valid_year,
        "n_seeds": args.n_seeds,
        "iterations": args.iterations,
        "depth": args.depth,
        "half_life": args.half_life,
        "standalone_bss": float(bss(fold_target, prediction)),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(fold_target.mean()),
        "baselines": baselines,
        "best_robust": ranking[:30],
        "best_score": score_ranking[:30],
    }
    output = ROOT / "research" / (
        f"v43_multiclass_hl{args.half_life:g}_s{args.n_seeds}_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        prediction=prediction.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
