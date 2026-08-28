"""Train and freeze the roster-robust v54 command ensemble over v38."""
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
from research_inferred_pitch_priors import PITCH_TYPES, bss, reconstruct_labels
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
COMMAND_WEIGHT = .10
OVERLAP_SCALE = .45
OVERLAP_WEIGHT = .075
RECENT_WEIGHT = .05
JOINT_WEIGHT = .01875
HALF_LIFE = 2.
RECENT_SEEDS = (4801, 4902, 5003, 4966, 5067, 5168)
JOINT_SEEDS = (7200, 7301, 7402)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def common(args, seed, loss, *, iterations, learning_rate, depth,
           l2_leaf_reg, random_strength, bagging_temperature):
    result = dict(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        l2_leaf_reg=l2_leaf_reg, random_strength=random_strength,
        bagging_temperature=bagging_temperature, border_count=32,
        bootstrap_type="Bayesian", loss_function=loss, eval_metric=loss,
        random_seed=seed, task_type=args.task_type,
        thread_count=args.threads, allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, gpu_ram_part=.90)
    return result


def command_parameters(args):
    return common(
        args, 4300, "MultiClass", iterations=1300, learning_rate=.018,
        depth=7, l2_leaf_reg=400., random_strength=2.8,
        bagging_temperature=.7,
    )


def overlap_parameters(args):
    return common(
        args, 4500, "Logloss", iterations=1600,
        learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998,
        random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515,
    )


def recent_parameters(args, seed):
    return common(
        args, seed, "MultiClass", iterations=1000, learning_rate=.025,
        depth=6, l2_leaf_reg=160., random_strength=2.,
        bagging_temperature=.8,
    )


def joint_parameters(args, seed):
    return common(
        args, seed, "MultiClass", iterations=1200, learning_rate=.018,
        depth=7, l2_leaf_reg=420., random_strength=2.8,
        bagging_temperature=.7,
    )


def score(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    regular = game_type == "R"
    result["R"] = float(bss(target[regular], prediction[regular]))
    result["F"] = float(bss(target[~regular], prediction[~regular]))
    return result


def audited_oof():
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38_archive = {key: archive[key] for key in archive.files}
    active = v38_archive["season"] == 2024
    target = v38_archive["target"][active].astype(float)
    base = np.clip(v38_archive["blended"][active].astype(float), .005, .995)
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
    with np.load(ROOT / "research/v43_multiclass_hl2_s1_2024.npz") as archive:
        command = archive["prediction"].astype(float)
        game_type = archive["game_type"].astype(str)
    with np.load(ROOT / "research/v45_overlap_hl2_2024.npz") as archive:
        overlap = archive["overlap_probability"].astype(float)
    with np.load(ROOT / "research/v48_regime_command_s3_2024.npz") as archive:
        recent_a = archive["multiclass"].astype(float)
    with np.load(
        ROOT / "research/v49_regime_multiclass_complexity_2024.npz"
    ) as archive:
        recent_b = archive["d6_i1000"].astype(float)
    with np.load(
        ROOT / "research/v52_pitch_command_joint_s3_2024.npz"
    ) as archive:
        joint = archive["history_no_team"].astype(float)
    for name, value in {
        "failure": failure, "command": command, "overlap": overlap,
        "recent_a": recent_a, "recent_b": recent_b, "joint": joint,
    }.items():
        if len(value) != len(base):
            raise ValueError(f"{name} rows do not align with v38")

    futures = game_type == "F"
    corrected = np.clip(failure + OVERLAP_SCALE * overlap, .005, .995)
    prediction = base.copy()
    prediction[futures] = sigmoid(
        logit(base[futures])
        + COMMAND_WEIGHT * (logit(command[futures]) - logit(base[futures]))
        + OVERLAP_WEIGHT * (logit(corrected[futures]) - logit(base[futures]))
        + RECENT_WEIGHT * (
            logit(.5 * (recent_a[futures] + recent_b[futures]))
            - logit(base[futures])
        )
    )
    prediction = sigmoid(
        logit(prediction) + JOINT_WEIGHT * (logit(joint) - logit(base))
    )
    blocks = masks(len(base))
    baseline = score(target, base, blocks, game_type)
    scores = score(target, prediction, blocks, game_type)
    gains = {name: scores[name] - baseline[name] for name in scores}

    roster_path = ROOT / "research/v53_roster_stability.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster_report = roster["reports"]["anchor_joint_no_team_0.01875"]
    if (
        gains["all"] < 3.0
        or min(gains[f"q{i}"] for i in range(1, 5)) < 1.4
        or gains["R"] < 0.
        or roster_report["minimum_roster_gain"] < .8
        or roster_report["clustered_bootstrap"]["p05"] <= 0.
    ):
        raise RuntimeError(
            f"v54 promotion gate failed: gains={gains}, roster={roster_report}"
        )
    upgraded = v38_archive["blended"].astype(np.float64).copy()
    upgraded[active] = prediction
    np.savez_compressed(
        ROOT / "outputs/v54_oof_predictions.npz",
        **{key: value for key, value in v38_archive.items() if key != "blended"},
        blended=upgraded,
    )
    return {
        "baseline": baseline, "scores": scores, "gains": gains,
        "minimum_roster_gain": roster_report["minimum_roster_gain"],
        "pitcher_cluster_bootstrap": roster_report["clustered_bootstrap"],
    }


def main():
    args = arguments()
    model_dir = ROOT / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v38_lowcard_ensemble", "v54_roster_robust_command",
    ):
        raise ValueError(
            f"Expected v38/v54 artifacts, got {metadata.get('version')}"
        )

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
    overlap = (reverse & middle).astype(np.int8)
    command = np.full(len(raw), -1, dtype=np.int8)
    command[complete & (target == 1)] = 0
    command[complete & (target == 0) & reverse & ~middle] = 1
    command[complete & (target == 0) & ~reverse & middle] = 2
    command[complete & (target == 0) & reverse & middle] = 3
    command[complete & (target == 0) & ~reverse & ~middle] = 4

    pitch = labels["pitch_type"].map(
        {name: index for index, name in enumerate(PITCH_TYPES)}
    ).fillna(-1).to_numpy(np.int8)
    joint_complete = complete & (pitch >= 0)
    joint = np.where(
        joint_complete, 5 * pitch + command, -1,
    ).astype(np.int8)

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
    no_team_features = features.drop(columns=[
        "pitcher_team_id", "batter_team_id",
    ])
    no_team_categories = [
        column for column in LOW_CARD_CATEGORIES
        if column in no_team_features.columns
    ]

    seasons = raw["season"].to_numpy(np.int16)
    reference_season = int(seasons.max())
    sample_weight = np.exp(
        -np.log(2.) * (reference_season - seasons.astype(float)) / HALF_LIFE
    ).astype(np.float32)
    recent = (seasons >= 2023) & raw["game_type"].eq("F").to_numpy() & complete
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Training v54 coherent command model", flush=True)
    model = CatBoostClassifier(**command_parameters(args))
    model.fit(
        features.loc[complete], command[complete],
        sample_weight=sample_weight[complete],
        cat_features=list(LOW_CARD_CATEGORIES),
    )
    model.save_model(str(model_dir / "catboost_v54_command.cbm"))

    print("Training v54 overlap model", flush=True)
    model = CatBoostClassifier(**overlap_parameters(args))
    model.fit(
        features.loc[complete], overlap[complete],
        sample_weight=sample_weight[complete],
        cat_features=list(LOW_CARD_CATEGORIES),
    )
    model.save_model(str(model_dir / "catboost_v54_overlap.cbm"))

    for index, seed in enumerate(RECENT_SEEDS):
        print(f"Training v54 recent F model {index + 1}/{len(RECENT_SEEDS)}", flush=True)
        model = CatBoostClassifier(**recent_parameters(args, seed))
        model.fit(
            features.loc[recent], command[recent],
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        model.save_model(str(model_dir / f"catboost_v54_recent_{index}.cbm"))

    for index, seed in enumerate(JOINT_SEEDS):
        print(f"Training v54 roster-robust joint model {index + 1}/{len(JOINT_SEEDS)}", flush=True)
        model = CatBoostClassifier(**joint_parameters(args, seed))
        model.fit(
            no_team_features.loc[joint_complete], joint[joint_complete],
            sample_weight=sample_weight[joint_complete],
            cat_features=no_team_categories,
        )
        model.save_model(str(model_dir / f"catboost_v54_joint_{index}.cbm"))

    report = audited_oof()
    names = [
        name for name in metadata.get("model_names", [])
        if name not in ("v25_temporal_portfolio", "v26_pareto_portfolio")
    ]
    if "v54_roster_robust_command" not in names:
        names.append("v54_roster_robust_command")
    metadata["model_names"] = names
    metadata["version"] = "v54_roster_robust_command"
    metadata["v54_roster_robust_command"] = {
        "categorical_columns": list(LOW_CARD_CATEGORIES),
        "feature_columns": list(features.columns),
        "joint_feature_columns": list(no_team_features.columns),
        "joint_categorical_columns": no_team_categories,
        "command_weight": COMMAND_WEIGHT,
        "overlap_scale": OVERLAP_SCALE,
        "overlap_weight": OVERLAP_WEIGHT,
        "recent_weight": RECENT_WEIGHT,
        "joint_weight": JOINT_WEIGHT,
        "recent_model_count": len(RECENT_SEEDS),
        "joint_model_count": len(JOINT_SEEDS),
        "recent_training_seasons": [2023, 2024],
        "raw_player_ids_used": False,
        "raw_team_ids_used_by_joint_model": False,
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
    }
    metadata.setdefault("training_info", {})["v54_validation"] = report
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(f"v54 validation: {json.dumps(report)}", flush=True)
    print("Stored v54 models, metadata, and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
