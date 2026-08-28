"""Predict fine pitch family and command outcome as a 40-class latent target.

The aligned Trackman pitch type is used only to construct historical training
targets.  Validation and inference features never include the current pitch
type, Trackman measurements, raw player IDs, or team IDs.
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
from research_v56_fine_pitch_failure_prior import PITCH_TYPES, fine_history


ROOT = Path(__file__).resolve().parent
COMMAND_NAMES = (
    "success", "reverse_only", "middle_only", "reverse_middle", "wayoff",
)
LOW_CARD_CATEGORIES = ("base_state", "game_dayofweek")
DROP_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "team_matchup",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed):
    result = dict(
        iterations=args.iterations, learning_rate=.02, depth=6,
        l2_leaf_reg=450., random_strength=2.8, bagging_temperature=.7,
        border_count=32, bootstrap_type="Bayesian",
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=5700 + seed, task_type=args.task_type,
        thread_count=args.threads, allow_writing_files=False, verbose=100,
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


def joint_target(raw, history):
    labels = history.set_index("row_id")[[
        "pitch_type_fine", "control_success", "reverse", "middle", "wayoff",
    ]]
    row_ids = raw["row_id"].astype(str)
    aligned = row_ids.isin(labels.index).to_numpy()
    local = labels.reindex(row_ids).reset_index(drop=True)
    command = np.full(len(raw), -1, dtype=np.int8)
    success = local["control_success"].eq(1).to_numpy()
    reverse = local["reverse"].eq(1).to_numpy()
    middle = local["middle"].eq(1).to_numpy()
    wayoff = local["wayoff"].eq(1).to_numpy()
    command[aligned & success] = 0
    command[aligned & ~success & reverse & ~middle] = 1
    command[aligned & ~success & ~reverse & middle] = 2
    command[aligned & ~success & reverse & middle] = 3
    command[aligned & ~success & wayoff] = 4
    pitch = local["pitch_type_fine"].map(
        {name: index for index, name in enumerate(PITCH_TYPES)}
    ).fillna(-1).to_numpy(np.int8)
    complete = aligned & (command >= 0) & (pitch >= 0)
    joint = np.where(complete, 5 * pitch + command, -1).astype(np.int8)
    return joint, complete


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    history = fine_history(full, labels)
    joint, complete = joint_target(raw, history)

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    features = features.drop(columns=[
        column for column in DROP_COLUMNS if column in features
    ])
    for column in LOW_CARD_CATEGORIES:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = (seasons < args.valid_year) & complete
    valid = seasons == args.valid_year
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    class_counts = {
        f"{pitch}_{command}": int(np.sum(
            joint[train] == 5 * pitch_index + command_index
        ))
        for pitch_index, pitch in enumerate(PITCH_TYPES)
        for command_index, command in enumerate(COMMAND_NAMES)
    }
    print(json.dumps({
        "valid_year": args.valid_year, "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()), "features": int(features.shape[1]),
        "classes_present": int(len(np.unique(joint[train]))),
        "raw_player_ids_used": False, "team_ids_used": False,
        "current_pitch_type_used": False, "forbidden_2025_trackman_used": False,
    }), flush=True)

    members = []
    for seed_index in range(args.n_seeds):
        model = CatBoostClassifier(**parameters(args, 101 * seed_index))
        model.fit(
            features.loc[train], joint[train], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        probabilities = model.predict_proba(features.loc[valid])
        classes = np.asarray(model.classes_, dtype=int)
        members.append(probabilities[:, classes % 5 == 0].sum(axis=1))
        print(f"fine-joint seed={seed_index + 1}/{args.n_seeds} complete", flush=True)
    prediction = np.mean(members, axis=0)

    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v54 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    direction = logit(np.clip(prediction, .005, .995)) - logit(base)
    reports = []
    for gate in ("all", "R", "F"):
        selected = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
        for weight in np.round(np.arange(-.10, .301, .0125), 4):
            candidate = base.copy()
            candidate[selected] = sigmoid(
                logit(base[selected]) + weight * direction[selected]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {name: result[name] - baseline[name] for name in result}
            reports.append({
                "gate": gate, "weight": float(weight), "scores": result,
                "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["R"], row["gains"]["F"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "baseline": baseline,
        "class_counts": class_counts,
        "standalone": {
            "all": float(bss(fold_target, prediction)),
            "R": float(bss(fold_target[game_type == "R"], prediction[game_type == "R"])),
            "F": float(bss(fold_target[game_type == "F"], prediction[game_type == "F"])),
            "mean": float(prediction.mean()),
            "correlation_base": float(np.corrcoef(prediction, base)[0, 1]),
        },
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:50],
        "best_score": sorted(
            reports, key=lambda row: row["scores"]["all"], reverse=True,
        )[:50],
        "raw_player_ids_used": False, "team_ids_used": False,
        "current_pitch_type_used": False, "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research" / f"v57_fine_pitch_joint_s{args.n_seeds}_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        base=base.astype(np.float32), prediction=prediction.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
