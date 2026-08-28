"""Train paired F-regime specialists with and without fraction confidence.

Both members use identical rows, targets, weights, categorical handling, and
random seeds.  Their prediction difference therefore targets the incremental
information in row-local recent-window fraction features.
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

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from recent_window_features import recent_window_features
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed_index):
    result = dict(
        iterations=args.iterations, learning_rate=.018, depth=7,
        l2_leaf_reg=360., random_strength=2.8, bagging_temperature=1.2,
        border_count=32, bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", random_seed=5900 + 101 * seed_index,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, gpu_ram_part=.90)
    return result


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    recent = recent_window_features(raw)
    bases = training_history_arrays(raw, target_series)
    base_features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(base_features, raw)
    base_features = add_state_interactions(base_features)
    base_features = base_features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in base_features
    ])
    for column in LOW_CARD_CATEGORIES:
        base_features[column] = base_features[column].fillna(-1).astype(np.int32)
    extra_features = pd.concat([base_features, recent], axis=1)

    seasons = raw["season"].to_numpy(np.int16)
    futures = raw["game_type"].astype(str).eq("F").to_numpy()
    train = (seasons < args.valid_year) & futures
    valid = (seasons == args.valid_year) & futures
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    print(json.dumps({
        "valid_year": args.valid_year,
        "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "base_features": int(base_features.shape[1]),
        "extra_features": int(extra_features.shape[1]),
        "new_features": int(recent.shape[1]),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }), flush=True)

    base_members = []
    extra_members = []
    for seed_index in range(args.n_seeds):
        print(f"v59 paired seed={seed_index + 1}/{args.n_seeds}: base", flush=True)
        model = CatBoostClassifier(**parameters(args, seed_index))
        model.fit(
            base_features.loc[train], target[train], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        base_members.append(model.predict_proba(base_features.loc[valid])[:, 1])
        print(f"v59 paired seed={seed_index + 1}/{args.n_seeds}: fraction", flush=True)
        model = CatBoostClassifier(**parameters(args, seed_index))
        model.fit(
            extra_features.loc[train], target[train], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        extra_members.append(model.predict_proba(extra_features.loc[valid])[:, 1])

    reference = np.mean(base_members, axis=0)
    prediction = np.mean(extra_members, axis=0)
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        v54 = np.clip(archive["blended"][active].astype(float), .005, .995)
    year_target = target[seasons == args.valid_year]
    if not np.allclose(fold_target, year_target):
        raise ValueError("v54 and train.csv rows do not align")
    valid_in_year = raw.loc[seasons == args.valid_year, "game_type"].astype(str).eq("F").to_numpy()
    if not np.allclose(fold_target[valid_in_year], target[valid]):
        raise ValueError("F validation rows do not align")
    diagnostics = {
        "valid_year": args.valid_year, "n_seeds": args.n_seeds,
        "reference_bss_F": float(bss(target[valid], reference)),
        "fraction_bss_F": float(bss(target[valid], prediction)),
        "standalone_gain_F": float(
            bss(target[valid], prediction) - bss(target[valid], reference)
        ),
        "prediction_delta_correlation": float(np.corrcoef(reference, prediction)[0, 1]),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research" / f"v59_f_fraction_s{args.n_seeds}_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32), base=v54.astype(np.float32),
        valid_f=valid_in_year, reference=reference.astype(np.float32),
        prediction=prediction.astype(np.float32),
        reference_members=np.asarray(base_members, dtype=np.float32),
        prediction_members=np.asarray(extra_members, dtype=np.float32),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
