"""Train and freeze the v38 low-cardinality ensemble over the v24 base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from failure_context import freeze_prior_context, prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss, reconstruct_labels


LABELS = ("reverse", "middle", "wayoff")
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
FAILURE_WEIGHT = .175
DIRECT_WEIGHT = .10
HALF_LIFE = 2.
FAILURE_SEEDS = (340, 341, 342)
DIRECT_SEEDS = (350, 451, 552)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def failure_parameters(args, seed):
    result = dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", random_seed=seed,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def direct_parameters(args, seed):
    result = dict(
        iterations=1100, learning_rate=.015, depth=7,
        l2_leaf_reg=300., random_strength=2.8,
        bagging_temperature=1.4, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", random_seed=seed,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
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
    for index, part in enumerate(np.array_split(position, 4), 1):
        mask = np.zeros(length, dtype=bool)
        mask[part] = True
        result[f"q{index}"] = mask
    return result


def audited_oof(root, raw):
    with np.load(root / "outputs/v23_oof_predictions.npz") as archive:
        v23 = {key: archive[key] for key in archive.files}
    with np.load(root / "outputs/v24_oof_predictions.npz") as archive:
        v24 = {key: archive[key] for key in archive.files}
    with np.load(
        root / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = {key: archive[key] for key in archive.files}
    with np.load(
        root / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
    ) as archive:
        direct = {key: archive[key] for key in archive.files}

    active = v24["season"] == 2024
    target = v24["target"][active].astype(float)
    base = np.clip(v24["blended"][active].astype(float), .005, .995)
    public_anchor = np.clip(v23["blended"][active].astype(float), .005, .995)
    if not (
        np.allclose(target, failure["target"])
        and np.allclose(target, direct["target"])
        and np.allclose(target, raw.loc[raw["season"].eq(2024), TARGET_COL])
    ):
        raise ValueError("v23/v24/v34/v35/train rows do not align")
    prediction = sigmoid(
        (1. - FAILURE_WEIGHT) * logit(base)
        + FAILURE_WEIGHT * logit(failure["new_failure"])
    )
    prediction = sigmoid(
        (1. - DIRECT_WEIGHT) * logit(prediction)
        + DIRECT_WEIGHT * logit(direct["prediction"])
    )
    masks = block_masks(len(target))
    scores = {name: float(bss(target[mask], prediction[mask]))
              for name, mask in masks.items()}
    gains_v23 = {
        name: scores[name] - float(bss(target[mask], public_anchor[mask]))
        for name, mask in masks.items()
    }
    game_type = raw.loc[raw["season"].eq(2024), "game_type"].astype(str).to_numpy()
    reports = {
        "v23_bss": float(bss(target, public_anchor)),
        "v24_bss": float(bss(target, base)),
        "v38_bss": scores["all"],
        "scores": scores,
        "gains_v23": gains_v23,
        "gain_R_v23": float(
            bss(target[game_type == "R"], prediction[game_type == "R"])
            - bss(target[game_type == "R"], public_anchor[game_type == "R"])
        ),
        "gain_F_v23": float(
            bss(target[game_type == "F"], prediction[game_type == "F"])
            - bss(target[game_type == "F"], public_anchor[game_type == "F"])
        ),
    }
    if (
        reports["v38_bss"] < 1020.
        or min(gains_v23[f"q{i}"] for i in range(1, 5)) < 18.
        or min(gains_v23["h1"], gains_v23["h2"]) < 25.
    ):
        raise RuntimeError(f"v38 promotion gate failed: {reports}")
    upgraded = v24["blended"].astype(np.float64).copy()
    upgraded[active] = prediction
    output = root / "outputs/v38_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in v24.items() if key != "blended"},
        blended=upgraded,
    )
    return reports


def main():
    args = arguments()
    root = Path(__file__).resolve().parent
    model_dir = root / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v24_robust_command_resolution", "v25_strict_temporal_portfolio",
        "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble",
    ):
        raise ValueError(f"Expected v24/v25/v26 artifacts, got {metadata.get('version')}")

    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    bases = training_history_arrays(data, target_series)
    base_features = engineer_features(
        data, *bases, global_prior=float(target.mean()),
    )
    add_training_component_features(base_features, data)
    base_features = add_state_interactions(base_features)
    context = prior_season_context(full, labels)
    failure_features = pd.concat([base_features, context], axis=1)
    direct_features = base_features.copy()
    drop_columns = [
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in base_features
    ]
    failure_features = failure_features.drop(columns=drop_columns)
    direct_features = direct_features.drop(columns=drop_columns)
    for column in LOW_CARD_CATEGORIES:
        failure_features[column] = failure_features[column].fillna(-1).astype(np.int32)
        direct_features[column] = direct_features[column].fillna(-1).astype(np.int32)

    reference_season = int(data["season"].max())
    age = reference_season - data["season"].to_numpy(float)
    weights = np.exp(-np.log(2.) * age / HALF_LIFE).astype(np.float32)
    usable = labels[list(LABELS)].notna().all(axis=1).to_numpy()
    model_dir.mkdir(parents=True, exist_ok=True)
    for label, seed in zip(LABELS, FAILURE_SEEDS):
        print(f"Training v38 failure model: {label}", flush=True)
        model = CatBoostClassifier(**failure_parameters(args, seed))
        model.fit(
            failure_features.loc[usable], labels.loc[usable, label].to_numpy(np.int8),
            sample_weight=weights[usable], cat_features=list(LOW_CARD_CATEGORIES),
        )
        model.save_model(str(model_dir / f"catboost_v38_failure_{label}.cbm"))
    for index, seed in enumerate(DIRECT_SEEDS):
        print(f"Training v38 direct model: seed={seed}", flush=True)
        model = CatBoostClassifier(**direct_parameters(args, seed))
        model.fit(
            direct_features, target, sample_weight=weights,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        model.save_model(str(model_dir / f"catboost_v38_direct_{index}.cbm"))

    reports = audited_oof(root, full)
    names = [
        name for name in metadata.get("model_names", [])
        if name not in ("v25_temporal_portfolio", "v26_pareto_portfolio")
    ]
    if "v38_lowcard_ensemble" not in names:
        names.append("v38_lowcard_ensemble")
    metadata["model_names"] = names
    metadata["version"] = "v38_lowcard_ensemble"
    metadata["v38_lowcard_ensemble"] = {
        "labels": list(LABELS),
        "categorical_columns": list(LOW_CARD_CATEGORIES),
        "failure_feature_columns": list(failure_features.columns),
        "direct_feature_columns": list(direct_features.columns),
        "failure_context": freeze_prior_context(
            full, labels, reference_season + 1,
        ),
        "failure_weight": FAILURE_WEIGHT,
        "direct_weight": DIRECT_WEIGHT,
        "half_life": HALF_LIFE,
        "failure_seeds": list(FAILURE_SEEDS),
        "direct_seeds": list(DIRECT_SEEDS),
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
    }
    metadata.setdefault("training_info", {})["v38_validation"] = reports
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(f"v38 validation: {json.dumps(reports)}", flush=True)
    print("Stored v38 models, metadata, and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
