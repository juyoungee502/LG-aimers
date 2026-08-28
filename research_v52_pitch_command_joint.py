"""Jointly predict latent pitch family and coherent command outcome.

The current pitch type is never an inference feature.  Historical training
labels are reconstructed from the next within-season cumulative state, and a
15-class model learns pitch-family x command-outcome probabilities.  Success
is the sum of the three pitch-family success classes.
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
from research_inferred_pitch_priors import PITCH_TYPES, bss, reconstruct_labels
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
COMMAND_NAMES = (
    "success", "reverse_only", "middle_only", "reverse_middle", "wayoff",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument(
        "--modes", nargs="+", choices=(
            "history", "history_no_team", "recent_f", "recent_f_no_team",
        ),
        default=(
            "history", "history_no_team", "recent_f", "recent_f_no_team",
        ),
    )
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed, mode):
    recent = mode.startswith("recent_f")
    result = dict(
        iterations=args.iterations, learning_rate=.018,
        depth=6 if recent else 7,
        l2_leaf_reg=260. if recent else 420.,
        random_strength=2. if recent else 2.8,
        bagging_temperature=.7, border_count=32, bootstrap_type="Bayesian",
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=(
            5200 + seed + (1000 if recent else 0)
            + (2000 if mode.endswith("no_team") else 0)
        ),
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, gpu_ram_part=.90)
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
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)

    complete = labels[["reverse", "middle", "pitch_type"]].notna().all(axis=1).to_numpy()
    reverse = labels["reverse"].fillna(0).eq(1).to_numpy()
    middle = labels["middle"].fillna(0).eq(1).to_numpy()
    command = np.full(len(raw), -1, dtype=np.int8)
    command[complete & (target == 1)] = 0
    command[complete & (target == 0) & reverse & ~middle] = 1
    command[complete & (target == 0) & ~reverse & middle] = 2
    command[complete & (target == 0) & reverse & middle] = 3
    command[complete & (target == 0) & ~reverse & ~middle] = 4
    pitch = labels["pitch_type"].map(
        {name: index for index, name in enumerate(PITCH_TYPES)}
    ).fillna(-1).to_numpy(np.int8)
    joint = np.where(complete, 5 * pitch + command, -1).astype(np.int8)

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
    game_type_all = raw["game_type"].astype(str).to_numpy()
    valid = seasons == args.valid_year
    game_type = game_type_all[valid]
    definitions = {
        "history": (seasons < args.valid_year) & complete,
        "history_no_team": (seasons < args.valid_year) & complete,
        "recent_f": (
            (seasons == args.valid_year - 1)
            & (game_type_all == "F") & complete
        ),
        "recent_f_no_team": (
            (seasons == args.valid_year - 1)
            & (game_type_all == "F") & complete
        ),
    }
    predictions = {}
    class_diagnostics = {}
    for mode in args.modes:
        train = definitions[mode]
        model_features = features
        categorical_columns = list(LOW_CARD_CATEGORIES)
        if mode.endswith("no_team"):
            model_features = features.drop(columns=[
                "pitcher_team_id", "batter_team_id",
            ])
            categorical_columns = [
                column for column in categorical_columns
                if column in model_features.columns
            ]
        age = (args.valid_year - 1) - seasons[train].astype(float)
        sample_weight = (
            np.exp(-np.log(2.) * age / 2.).astype(np.float32)
            if mode.startswith("history") else None
        )
        members = []
        for seed_index in range(args.n_seeds):
            print(json.dumps({
                "mode": mode, "seed": seed_index + 1,
                "train_rows": int(train.sum()),
                "features": int(model_features.shape[1]),
            }), flush=True)
            model = CatBoostClassifier(**parameters(args, 101 * seed_index, mode))
            model.fit(
                model_features.loc[train], joint[train], sample_weight=sample_weight,
                cat_features=categorical_columns,
            )
            probabilities = model.predict_proba(model_features.loc[valid])
            classes = np.asarray(model.classes_, dtype=int)
            success_columns = [
                index for index, label in enumerate(classes) if label % 5 == 0
            ]
            members.append(probabilities[:, success_columns].sum(axis=1))
        predictions[mode] = np.mean(members, axis=0)
        class_diagnostics[mode] = {
            f"{PITCH_TYPES[pitch_index]}_{COMMAND_NAMES[command_index]}": int(
                np.sum(joint[train] == 5 * pitch_index + command_index)
            )
            for pitch_index in range(3) for command_index in range(5)
        }

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v38 rows do not align")
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    reports = []
    for name, value in predictions.items():
        direction = logit(np.clip(value, .005, .995)) - logit(base)
        for gate in ("all", "R", "F"):
            selected = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(-.10, .301, .0125), 4):
                candidate = base.copy()
                candidate[selected] = sigmoid(
                    logit(base[selected]) + weight * direction[selected]
                )
                result = score(fold_target, candidate, blocks, game_type)
                gains = {key: result[key] - baseline[key] for key in result}
                reports.append({
                    "name": name, "gate": gate, "weight": float(weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["R"], row["gains"]["F"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "n_seeds": args.n_seeds,
        "baseline": baseline, "class_counts": class_diagnostics,
        "standalone": {
            name: {
                "all": float(bss(fold_target, value)),
                "R": float(bss(fold_target[game_type == "R"], value[game_type == "R"])),
                "F": float(bss(fold_target[game_type == "F"], value[game_type == "F"])),
                "mean_F": float(value[game_type == "F"].mean()),
            } for name, value in predictions.items()
        },
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:50],
        "best_score": sorted(
            reports, key=lambda row: row["scores"]["all"], reverse=True,
        )[:50],
    }
    output = ROOT / "research" / f"v52_pitch_command_joint_s{args.n_seeds}_{args.valid_year}.npz"
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
