"""Screen a recency-weighted, low-cardinality native CatBoost classifier.

Player IDs are deliberately excluded.  Their stable as-of rates remain, while
base state, teams, and weekday are passed to CatBoost as actual categories.
The resulting model is evaluated only as a diverse blend over the final v23
OOF prediction, with separate all/R/F gates and time-block diagnostics.
"""
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


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1100)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(args, seed):
    result = dict(
        iterations=args.iterations,
        learning_rate=.015,
        depth=7,
        l2_leaf_reg=300.,
        random_strength=2.8,
        bagging_temperature=1.4,
        border_count=32,
        bootstrap_type="Bayesian",
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=350 + 101 * seed,
        task_type=args.task_type,
        thread_count=args.threads,
        allow_writing_files=False,
        verbose=100,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def masks(length):
    position = np.arange(length)
    result = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index, block in enumerate(np.array_split(position, 4), 1):
        mask = np.zeros(length, dtype=bool)
        mask[block] = True
        result[f"q{index}"] = mask
    return result


def gain_report(target, base, candidate, blocks):
    return {
        name: float(bss(target[mask], candidate[mask]) - bss(target[mask], base[mask]))
        for name, mask in blocks.items()
    }


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])
    missing = [column for column in LOW_CARD_CATEGORIES if column not in features]
    if missing:
        raise ValueError(f"Missing intended categorical columns: {missing}")
    for column in LOW_CARD_CATEGORIES:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    sample_weight = None
    if args.half_life > 0.:
        age = (args.valid_year - 1) - seasons[train].astype(float)
        sample_weight = np.exp(-np.log(2.) * age / args.half_life).astype(np.float32)

    seed_predictions = []
    for seed in range(args.n_seeds):
        print(
            f"Training v35 direct CatBoost seed {seed + 1}/{args.n_seeds}: "
            f"valid={args.valid_year}, rows={int(train.sum()):,}, "
            f"features={features.shape[1]}", flush=True,
        )
        model = CatBoostClassifier(**parameters(args, seed))
        model.fit(
            features.loc[train], target[train],
            sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        seed_predictions.append(model.predict_proba(features.loc[valid])[:, 1])
    prediction = np.mean(seed_predictions, axis=0)

    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        v23 = {key: archive[key] for key in archive.files}
    fold = v23["season"] == args.valid_year
    fold_target = v23["target"][fold].astype(float)
    base = np.clip(v23["blended"][fold].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v23 and train.csv validation rows do not align")

    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(fold_target))
    reports = []
    direction = logit(prediction) - logit(base)
    for gate in ("all", "R", "F"):
        active = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
        for scale in np.round(np.arange(0., .5001, .025), 4):
            candidate = base.copy()
            candidate[active] = sigmoid(logit(base[active]) + scale * direction[active])
            gains = gain_report(fold_target, base, candidate, blocks)
            reports.append({
                "gate": gate,
                "scale": float(scale),
                "gains": gains,
                "min_half": float(min(gains["h1"], gains["h2"])),
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "gain_R": float(
                    bss(fold_target[game_type == "R"], candidate[game_type == "R"])
                    - bss(fold_target[game_type == "R"], base[game_type == "R"])
                ),
                "gain_F": float(
                    bss(fold_target[game_type == "F"], candidate[game_type == "F"])
                    - bss(fold_target[game_type == "F"], base[game_type == "F"])
                ),
            })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    diagnostics = {
        "valid_year": args.valid_year,
        "n_seeds": args.n_seeds,
        "iterations": args.iterations,
        "half_life": args.half_life,
        "features": int(features.shape[1]),
        "categorical_columns": list(LOW_CARD_CATEGORIES),
        "model_bss": float(bss(fold_target, prediction)),
        "base_bss": float(bss(fold_target, base)),
        "correlation_model_base": float(np.corrcoef(prediction, base)[0, 1]),
        "top": reports[:50],
    }
    weight_tag = "uniform" if args.half_life <= 0. else f"hl{args.half_life:g}"
    output = ROOT / "research" / (
        f"v35_lowcard_direct_{weight_tag}_s{args.n_seeds}_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        base=base.astype(np.float32),
        prediction=prediction.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
