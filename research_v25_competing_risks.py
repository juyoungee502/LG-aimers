"""Structured competing-risk model for the three command-failure mechanisms.

Reverse and middle labels overlap, so subtracting their marginal probabilities
double-counts some failures.  This model uses conditional hazards instead:

  P(success) = (1-P(reverse))
               * (1-P(middle | not reverse))
               * (1-P(wayoff | neither))

The detailed labels are reconstructed from train rows only and are never used
at inference.  All input features remain pre-pitch and row independent.
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
from trackman_context import attach_context, pitcher_mapping, prepare_trackman
from v24_robust_candidate import time_safe_command_features


ROOT = Path(__file__).resolve().parent
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--seed", type=int, default=6200)
    return parser.parse_args()


def logit(probability):
    p = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(p / (1. - p))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def model(seed, multiclass=False):
    loss = "MultiClass" if multiclass else "Logloss"
    return CatBoostClassifier(
        iterations=1200, depth=6, learning_rate=.02, l2_leaf_reg=150.,
        random_strength=1., border_count=32, bootstrap_type="Bayesian",
        bagging_temperature=.5, max_ctr_complexity=1, one_hot_max_size=32,
        loss_function=loss, eval_metric=loss, task_type="GPU", devices="0",
        random_seed=seed, allow_writing_files=False, verbose=0,
    )


def recency_weight(seasons, reference):
    age = np.maximum(0., reference - seasons.astype(float))
    return np.exp(-np.log(2.) * age / 3.).astype(np.float32)


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def evaluate(y, base, specialist, rows, name):
    regular = rows["game_type"].eq("R").to_numpy()
    reports = []
    for weight in np.arange(-.15, .401, .01):
        candidate = base.copy()
        candidate[regular] = sigmoid(
            (1. - weight) * logit(base[regular])
            + weight * logit(specialist[regular])
        )
        values = {
            label: bss(y[active], candidate[active]) - bss(y[active], base[active])
            for label, active in masks(rows).items() if active.any()
        }
        reports.append({
            "name": name, "weight": float(weight), "gains": values,
            "min_half": min(values["half_1"], values["half_2"]),
            "min_quarter": min(values[f"q{i}"] for i in range(1, 5)),
            "min_month": min(values[key] for key in ("months_3_5", "months_6_7", "months_8_11")),
            "standalone_r_bss": bss(y[regular], specialist[regular]),
        })
    return reports


def main():
    args = arguments()
    data = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    usable = labels[["reverse", "middle", "wayoff"]].notna().all(axis=1).to_numpy()

    trackman = pd.read_csv(
        ROOT / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(ROOT, data, trackman)
    trackman_features = attach_context(data, prepare_trackman(trackman, mapping))
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    failure_features = prior_season_context(full, labels)
    command = time_safe_command_features(data, target)
    recent_command = time_safe_command_features(data, target, history_window=1).rename(
        columns=lambda column: f"recent_{column}"
    )
    features = pd.concat(
        [features, trackman_features, failure_features, command, recent_command], axis=1,
    ).drop(columns=["game_month", "season"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    regular = data["game_type"].eq("R").to_numpy()
    train = (seasons < args.valid_year) & regular & usable
    valid = seasons == args.valid_year
    weights = recency_weight(seasons[train], args.valid_year - 1)

    reverse = labels["reverse"].fillna(0).to_numpy(np.int8)
    middle = labels["middle"].fillna(0).to_numpy(np.int8)
    wayoff = labels["wayoff"].fillna(0).to_numpy(np.int8)
    hazard_predictions = []
    hazard_specs = (
        ("reverse", train, reverse),
        ("middle_given_no_reverse", train & (reverse == 0), middle),
        ("wayoff_given_neither", train & (reverse == 0) & (middle == 0), wayoff),
    )
    for offset, (name, active, label) in enumerate(hazard_specs):
        classifier = model(args.seed + offset)
        active_weight = recency_weight(seasons[active], args.valid_year - 1)
        classifier.fit(
            features.loc[active], label[active], sample_weight=active_weight,
            cat_features=CAT_COLUMNS,
        )
        hazard_predictions.append(classifier.predict_proba(features.loc[valid])[:, 1])
        print(f"completed {name}: train_rows={active.sum()}", flush=True)
    hazards = np.column_stack(hazard_predictions)
    competing = np.prod(1. - hazards, axis=1)

    priority = np.full(len(data), 3, dtype=np.int8)
    priority[wayoff == 1] = 2
    priority[middle == 1] = 1
    priority[reverse == 1] = 0
    multiclass = model(args.seed + 10, multiclass=True)
    multiclass.fit(
        features.loc[train], priority[train], sample_weight=weights,
        cat_features=CAT_COLUMNS,
    )
    probability = multiclass.predict_proba(features.loc[valid])
    success_position = list(multiclass.classes_.astype(int)).index(3)
    priority_success = probability[:, success_position]
    print(f"completed priority multiclass: train_rows={train.sum()}", flush=True)

    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        fold = archive["season"] == args.valid_year
        y = archive["target"][fold].astype(float)
        base = archive["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v24 OOF rows do not align")
    rows = data.loc[valid].reset_index(drop=True)
    reports = []
    reports.extend(evaluate(y, base, competing, rows, "conditional_hazards"))
    reports.extend(evaluate(y, base, priority_success, rows, "priority_multiclass"))
    reports.sort(
        key=lambda item: (
            min(item["min_half"], item["min_quarter"], item["min_month"]),
            item["gains"]["all"],
        ), reverse=True,
    )
    output = ROOT / f"research/v25_competing_risks_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        hazards=hazards.astype(np.float32), competing=competing.astype(np.float32),
        priority=priority_success.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "year": args.valid_year, "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "usable_rate": float(usable.mean()), "top": reports[:40],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
