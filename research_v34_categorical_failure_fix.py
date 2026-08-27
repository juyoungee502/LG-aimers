"""Test correct native-categorical plumbing in the failure decomposition.

The legacy failure screen declared categorical columns but did not pass them
to CatBoost.  This script changes that single modelling assumption, predicts a
strict future season, and audits both an additive blend and replacement of the
legacy failure direction over the final v23 OOF base.
"""
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


ROOT = Path(__file__).resolve().parent
LABELS = ("reverse", "middle", "wayoff")
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
LEGACY_VARIANT = "uniform_depth8"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument(
        "--profile", choices=("category_fix", "lowcard_no_ids"),
        default="category_fix",
    )
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument(
        "--half-life", type=float, default=0.,
        help="Season half-life; zero preserves the legacy uniform weighting.",
    )
    return parser.parse_args()


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def model_parameters(args, seed):
    result = dict(
        iterations=args.iterations,
        learning_rate=.01631820635235777,
        depth=8,
        l2_leaf_reg=509.6419153575998,
        random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515,
        border_count=32,
        bootstrap_type="Bayesian",
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type=args.task_type,
        thread_count=args.threads,
        random_seed=340 + seed,
        allow_writing_files=False,
        verbose=100,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def block_masks(length):
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


def gain_report(target, base, candidate, masks):
    return {
        name: float(bss(target[mask], candidate[mask]) - bss(target[mask], base[mask]))
        for name, mask in masks.items()
    }


def candidate_rows(target, base, new_failure, old_failure, regular):
    masks = block_masks(len(target))
    reports = []
    for mode, direction in (
        ("add", logit(new_failure) - logit(base)),
        ("replace", logit(new_failure) - logit(old_failure)),
    ):
        for scale in np.round(np.arange(0., .5001, .025), 4):
            candidate = np.asarray(base, dtype=float).copy()
            candidate[regular] = sigmoid(
                logit(base[regular]) + scale * direction[regular]
            )
            gains = gain_report(target, base, candidate, masks)
            reports.append({
                "mode": mode,
                "scale": float(scale),
                "gains": gains,
                "min_half": float(min(gains["h1"], gains["h2"])),
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
            })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    return reports


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)

    missing = [column for column in LOW_CARD_CATEGORIES if column not in features]
    if missing:
        raise ValueError(f"Missing intended categorical columns: {missing}")
    if args.profile == "lowcard_no_ids":
        features = features.drop(
            columns=[
                column for column in ("pitcher_id", "batter_id", "team_matchup")
                if column in features
            ]
        )
    for column in LOW_CARD_CATEGORIES:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    usable = labels[list(LABELS)].notna().all(axis=1).to_numpy() & train
    sample_weight = None
    if args.half_life > 0.:
        age = (args.valid_year - 1) - seasons[usable].astype(float)
        sample_weight = np.exp(-np.log(2.) * age / args.half_life).astype(np.float32)

    predictions = {}
    for offset, label in enumerate(LABELS):
        print(
            f"Training v34 {args.profile} {label}: valid={args.valid_year}, "
            f"rows={int(usable.sum()):,}, features={features.shape[1]}",
            flush=True,
        )
        seed_predictions = []
        for seed_index in range(args.n_seeds):
            model = CatBoostClassifier(**model_parameters(
                args, offset + 101 * seed_index,
            ))
            model.fit(
                features.loc[usable], labels.loc[usable, label].to_numpy(np.int8),
                sample_weight=sample_weight,
                cat_features=list(LOW_CARD_CATEGORIES),
            )
            seed_predictions.append(model.predict_proba(features.loc[valid])[:, 1])
        predictions[label] = np.mean(seed_predictions, axis=0)

    new_failure = np.clip(
        1. - sum(predictions[label] for label in LABELS), 1e-5, 1. - 1e-5,
    )
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        v23 = {key: archive[key] for key in archive.files}
    fold = v23["season"] == args.valid_year
    fold_target = v23["target"][fold].astype(float)
    base = np.clip(v23["blended"][fold].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v23 and train.csv validation rows do not align")

    legacy_path = ROOT / "research" / (
        f"failure_specialists_{args.valid_year}_prior_context.npz"
    )
    with np.load(legacy_path) as archive:
        variants = archive["variants"].astype(str).tolist()
        legacy = archive["predictions"][variants.index(LEGACY_VARIANT), :, :3]
    old_failure = np.clip(1. - legacy.sum(axis=1), 1e-5, 1. - 1e-5)
    regular = raw.loc[valid, "game_type"].eq("R").to_numpy()
    reports = candidate_rows(
        fold_target, base, new_failure, old_failure, regular,
    )

    diagnostics = {
        "valid_year": args.valid_year,
        "profile": args.profile,
        "iterations": args.iterations,
        "n_seeds": args.n_seeds,
        "half_life": args.half_life,
        "features": int(features.shape[1]),
        "categorical_columns": list(LOW_CARD_CATEGORIES),
        "removed_numeric_ids": args.profile == "lowcard_no_ids",
        "new_failure_bss": float(bss(fold_target, new_failure)),
        "old_failure_bss": float(bss(fold_target, old_failure)),
        "base_bss": float(bss(fold_target, base)),
        "correlation_new_old": float(np.corrcoef(new_failure, old_failure)[0, 1]),
        "correlation_new_base": float(np.corrcoef(new_failure, base)[0, 1]),
        "top": reports[:40],
    }
    weight_tag = "uniform" if args.half_life <= 0. else f"hl{args.half_life:g}"
    seed_tag = "" if args.n_seeds == 1 else f"_s{args.n_seeds}"
    output = ROOT / "research" / (
        f"v34_categorical_failure_{args.profile}_{weight_tag}"
        f"{seed_tag}_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        base=base.astype(np.float32),
        new_failure=new_failure.astype(np.float32),
        old_failure=old_failure.astype(np.float32),
        regular=regular,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
