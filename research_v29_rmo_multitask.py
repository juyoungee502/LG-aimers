"""Chronological auxiliary R/M/O failure-shape models for the v23 base.

The official FAQ explicitly permits auxiliary heads built from training-only
privileged labels, provided inference uses each test row independently.  This
screen reconstructs reverse/middle/outside labels only in train.csv, fits
models on seasons strictly before the validation year, and stores row-local
meta predictions for a separate residual-transfer audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
EPS = 1e-4
CAT_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def recover_labels(raw: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Recover train-only failure shapes from the next official as-of count."""
    work = raw[[
        "row_id", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    ]].copy()
    work["_row_number"] = pd.to_numeric(
        work["row_id"].str.extract(r"(\d+)", expand=False), errors="raise",
    )
    work["_original"] = np.arange(len(work), dtype=np.int64)
    work["_reverse_count"] = (
        work["asof_pitcher_reverse_rate"] * work["asof_pitcher_n"]
    )
    work["_middle_count"] = (
        work["asof_pitcher_middle_rate"] * work["asof_pitcher_n"]
    )
    work.sort_values("_row_number", inplace=True)
    grouped = work.groupby("pitcher_id", observed=True, sort=False)
    reverse = (
        grouped["_reverse_count"].shift(-1) - work["_reverse_count"]
    ).round()
    middle = (
        grouped["_middle_count"].shift(-1) - work["_middle_count"]
    ).round()
    outside = 1.0 - target[work["_original"].to_numpy()] - reverse - middle
    labels = np.full(len(work), -1, dtype=np.int8)
    success = target[work["_original"].to_numpy()] == 1
    labels[success & reverse.notna().to_numpy()] = 0
    labels[(~success) & reverse.eq(1).to_numpy()] = 1
    labels[(~success) & reverse.eq(0).to_numpy() & middle.eq(1).to_numpy()] = 2
    labels[(~success) & reverse.eq(0).to_numpy() & middle.eq(0).to_numpy()
           & outside.eq(1).to_numpy()] = 3
    restored = np.full(len(work), -1, dtype=np.int8)
    restored[work["_original"].to_numpy()] = labels
    return restored


def parameters(args, seed, multiclass=False):
    result = dict(
        iterations=800 if args.full else 250,
        learning_rate=.03 if args.full else .05,
        depth=6,
        l2_leaf_reg=100.,
        random_strength=1.,
        border_count=32,
        bootstrap_type="Bayesian",
        bagging_temperature=.5,
        loss_function="MultiClass" if multiclass else "Logloss",
        task_type=args.task_type,
        devices=args.devices if args.task_type == "GPU" else None,
        random_seed=seed,
        allow_writing_files=False,
        verbose=100,
        thread_count=args.threads,
    )
    if args.task_type == "CPU":
        result.pop("devices")
    return result


def fit_binary(features, label, train, valid_rows, args, seed):
    model = CatBoostClassifier(**parameters(args, seed))
    model.fit(features.iloc[np.flatnonzero(train)], label[train],
              cat_features=list(CAT_COLUMNS))
    return model.predict_proba(features.iloc[valid_rows])[:, 1]


def segment_gains(target, base, candidate, rows):
    position = np.arange(len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }
    return {
        name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
        for name, mask in masks.items() if mask.any()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True,
                        choices=(2022, 2023, 2024))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig",
                      low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    labels = recover_labels(raw, target)
    valid_labels = labels >= 0
    counts = np.bincount(labels[valid_labels], minlength=4)
    print(f"RMO recovered rows={valid_labels.sum()} counts={counts.tolist()}",
          flush=True)

    history = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *history, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features).copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    source = (seasons < args.valid_year) & valid_labels
    valid = seasons == args.valid_year
    valid_rows = np.flatnonzero(valid)
    print(f"RMO year={args.valid_year} source={source.sum()} valid={valid.sum()}",
          flush=True)

    q_reverse = fit_binary(
        features, (labels == 1).astype(np.int8), source, valid_rows, args, 7201,
    )
    not_reverse = source & (labels != 1)
    q_middle = fit_binary(
        features, (labels == 2).astype(np.int8), not_reverse, valid_rows,
        args, 7202,
    )
    not_reverse_middle = source & (labels != 1) & (labels != 2)
    q_outside = fit_binary(
        features, (labels == 3).astype(np.int8), not_reverse_middle,
        valid_rows, args, 7203,
    )
    joint = CatBoostClassifier(**parameters(args, 7204, multiclass=True))
    joint.fit(features.iloc[np.flatnonzero(source)], labels[source],
              cat_features=list(CAT_COLUMNS))
    joint_probability = joint.predict_proba(features.iloc[valid_rows])

    hazard_failure = (
        q_reverse
        + (1. - q_reverse) * q_middle
        + (1. - q_reverse) * (1. - q_middle) * q_outside
    )
    hazard_success = np.clip(1. - hazard_failure, EPS, 1. - EPS)
    joint_success = np.clip(joint_probability[:, 0], EPS, 1. - EPS)
    meta = np.column_stack([
        q_reverse, q_middle, q_outside,
        np.log((q_middle + EPS) / (q_reverse + EPS)),
        np.log((q_outside + EPS) / (q_reverse + EPS)),
        np.log((q_outside + EPS) / (q_middle + EPS)),
        *[joint_probability[:, index] for index in range(4)],
    ])
    names = np.asarray([
        "q_reverse", "q_middle", "q_outside", "log_middle_reverse",
        "log_outside_reverse", "log_outside_middle", "joint_success",
        "joint_reverse", "joint_middle", "joint_outside",
    ])

    reports = {
        "year": args.valid_year,
        "label_counts": counts.tolist(),
        "hazard_bss": bss(target[valid], hazard_success),
        "joint_bss": bss(target[valid], joint_success),
    }
    oof_path = ROOT / "outputs/v23_oof_predictions.npz"
    if args.valid_year in (2023, 2024) and oof_path.exists():
        with np.load(oof_path) as archive:
            fold = archive["season"] == args.valid_year
            base = archive["blended"][fold].astype(float)
            expected = archive["target"][fold].astype(float)
        if not np.allclose(expected, target[valid]):
            raise ValueError("v23 OOF rows do not align with raw validation rows")
        rows = raw.loc[valid].reset_index(drop=True)
        blends = []
        for model_name, alternative in (
            ("hazard", hazard_success), ("joint", joint_success),
        ):
            for weight in np.arange(-.10, .301, .01):
                candidate = np.clip((1. - weight) * base + weight * alternative,
                                    .005, .995)
                gains = segment_gains(expected, base, candidate, rows)
                blends.append({
                    "model": model_name, "weight": float(weight),
                    "gains": gains, "min_segment": min(gains.values()),
                })
        blends.sort(key=lambda row: (row["min_segment"], row["gains"]["all"]),
                    reverse=True)
        reports["v23_blend_top"] = blends[:30]

    output = ROOT / f"research/v29_rmo_multitask_{args.valid_year}.npz"
    np.savez_compressed(
        output, names=names, meta=meta.astype(np.float32),
        hazard_success=hazard_success.astype(np.float32),
        joint_success=joint_success.astype(np.float32),
        target=target[valid].astype(np.float32), season=seasons[valid],
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps(reports, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
