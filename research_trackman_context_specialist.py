"""Validate CatBoost specialists with historical Trackman context features."""
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
from trackman_context import attach_context, pitcher_mapping, prepare_trackman


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def model_parameters(seed, variant):
    if variant == "weighted_depth6":
        return dict(
            iterations=1200, learning_rate=.02, depth=6, l2_leaf_reg=100.0,
            random_strength=1.0, border_count=32,
        )
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    trackman_columns = [
        "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
        "balls_before", "strikes_before", "batter_hand", "rel_speed",
    ]
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv", usecols=trackman_columns,
        encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    context = attach_context(data, trackman)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    with np.load(root / "outputs" / "v16_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    oof_mask = oof["season"] == args.valid_year
    y = oof["target"][oof_mask].astype(float)
    base = oof["blended"][oof_mask].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("OOF rows do not align")
    reports, predictions = [], {}
    for offset, variant in enumerate(("weighted_depth6", "uniform_depth8")):
        sample_weight = None
        if variant == "weighted_depth6":
            age = (args.valid_year - 1) - seasons[train].astype(float)
            sample_weight = np.exp(-np.log(2.0) * age / 3.0).astype(np.float32)
        model = CatBoostClassifier(
            **model_parameters(610 + offset, variant), loss_function="Logloss",
            eval_metric="Logloss", task_type="GPU", devices="0",
            random_seed=610 + offset, allow_writing_files=False, verbose=0,
        )
        model.fit(
            features.loc[train], target[train], sample_weight=sample_weight,
            cat_features=CAT_COLUMNS,
        )
        prediction = model.predict_proba(features.loc[valid])[:, 1]
        predictions[variant] = prediction.astype(np.float32)
        half = len(y) // 2
        for weight in np.arange(0.0, .401, .025):
            blended = sigmoid((1.0 - weight) * logit(base) + weight * logit(prediction))
            values = [
                bss(y, blended) - bss(y, base),
                bss(y[:half], blended[:half]) - bss(y[:half], base[:half]),
                bss(y[half:], blended[half:]) - bss(y[half:], base[half:]),
            ]
            reports.append({
                "valid_year": args.valid_year, "variant": variant,
                "weight": float(weight), "gain": values[0],
                "gain_first_half": values[1], "gain_second_half": values[2],
                "min_half": min(values[1:]), "standalone_bss": bss(y, prediction),
            })
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research" / f"trackman_context_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        weighted_depth6=predictions["weighted_depth6"],
        uniform_depth8=predictions["uniform_depth8"],
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "mapped_pitchers": len(mapping),
        "mapped_confidence_min": float(mapping_report["confidence"].min()),
        "trackman_rows": len(trackman), "top": reports[:40],
    }, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
