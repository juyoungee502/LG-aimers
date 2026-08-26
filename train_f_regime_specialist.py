"""Promote the robust post-2022 F-regime CatBoost specialist to v18."""
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


SEEDS = (1812, 2025, 3407)
BLEND_WEIGHT = .35


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="submit/model")
    parser.add_argument("--diagnostic-dir", default="outputs")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(args, seed):
    result = dict(
        iterations=1000, learning_rate=.025, depth=6,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=100.,
        random_strength=1., random_seed=seed, border_count=32,
        allow_writing_files=False, verbose=0, task_type=args.task_type,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def fit_predict(features, target, train, valid, args, save_dir=None):
    predictions = []
    for index, seed in enumerate(SEEDS):
        model = CatBoostClassifier(**parameters(args, seed))
        model.fit(features.loc[train], target[train])
        if valid is not None:
            predictions.append(model.predict_proba(features.loc[valid])[:, 1])
        if save_dir is not None:
            model.save_model(str(save_dir / f"catboost_f_regime_{index}.cbm"))
        print(f"F-regime model complete: seed={seed}", flush=True)
    return np.mean(predictions, axis=0) if predictions else None


def main():
    args = arguments()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        Path(args.data_dir) / "train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v17_trackman_context":
        raise ValueError(f"Expected v17 metadata, got {metadata.get('version')}")
    for column in metadata["cat_features"]:
        features[column] = features[column].fillna(-1).astype(np.int32)
    if list(features.columns) != metadata["feature_columns"]:
        raise ValueError("F-regime feature order differs from base model")

    seasons = data["season"].to_numpy(np.int16)
    f_gate = data["game_type"].eq("F").to_numpy()
    train = (seasons == 2023) & f_gate
    valid = (seasons == 2024) & f_gate
    specialist = fit_predict(features, target, train, valid, args)

    diagnostic_dir = Path(args.diagnostic_dir)
    with np.load(diagnostic_dir / "v17_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    latest = oof["season"] == 2024
    latest_data = data.loc[seasons == 2024]
    latest_f = latest_data["game_type"].eq("F").to_numpy()
    valid_target = target[valid].astype(np.float64)
    if not np.allclose(valid_target, oof["target"][latest][latest_f]):
        raise ValueError("v17 OOF and F validation rows do not align")
    base_f = oof["blended"][latest][latest_f].astype(np.float64)
    blended_f = sigmoid(
        (1. - BLEND_WEIGHT) * logit(base_f)
        + BLEND_WEIGHT * logit(specialist)
    )
    midpoint = len(valid_target) // 2
    halves = [np.arange(len(valid_target)) < midpoint, np.arange(len(valid_target)) >= midpoint]
    half_gains = [
        bss(valid_target[mask], blended_f[mask]) - bss(valid_target[mask], base_f[mask])
        for mask in halves
    ]
    base_score = bss(valid_target, base_f)
    f_score = bss(valid_target, blended_f)
    if f_score <= base_score or min(half_gains) <= 0.:
        raise RuntimeError(
            f"F specialist failed promotion gate: gain={f_score-base_score}, halves={half_gains}"
        )

    upgraded = oof["blended"].astype(np.float64).copy()
    latest_positions = np.flatnonzero(latest)
    upgraded[latest_positions[latest_f]] = blended_f
    overall_base = bss(oof["target"][latest], oof["blended"][latest])
    overall_v18 = bss(oof["target"][latest], upgraded[latest])
    report = {
        "f_base_bss": base_score, "f_v18_bss": f_score,
        "f_gain": f_score - base_score, "half_gains": half_gains,
        "overall_v17_bss": overall_base, "overall_v18_bss": overall_v18,
        "overall_gain": overall_v18 - overall_base,
        "specialist_mean": float(specialist.mean()),
    }
    print(f"v18 validation: {json.dumps(report)}", flush=True)

    final_train = (seasons >= 2023) & f_gate
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_predict(features, target, final_train, None, args, save_dir=model_dir)
    metadata["version"] = "v18_f_regime"
    names = metadata.setdefault("model_names", [])
    if "f_regime_specialist" not in names:
        names.append("f_regime_specialist")
    metadata["f_regime"] = {
        "blend_weight": BLEND_WEIGHT, "game_type": "F",
        "model_feature_columns": list(features.columns),
        "training_seasons": [2023, 2024], "seeds": list(SEEDS),
    }
    metadata["training_info"]["v18_validation"] = report
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )

    specialist_full = np.full(len(upgraded), np.nan, dtype=np.float64)
    specialist_full[latest_positions[latest_f]] = specialist
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostic_dir / "v18_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded, f_specialist=specialist_full,
    )
    print("Stored v18 F-regime models and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
