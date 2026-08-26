"""Validate richer prior-season Trackman pitcher profiles on top of v17."""
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
from trackman_extended import attach_profiles, prepare_physical


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman_columns = [
        "trackman_id", "season", "pitcher_trackman_id", "pitcher_hand",
        "pitch_type_group", "balls_before", "strikes_before", "batter_hand",
        "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed",
    ]
    raw_trackman = pd.read_csv(
        root / "data/trackman_history.csv", usecols=trackman_columns,
        encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, raw_trackman)
    context_trackman = prepare_trackman(raw_trackman, mapping)
    physical_trackman = prepare_physical(raw_trackman, mapping)
    context = attach_context(data, context_trackman)
    profiles = attach_profiles(data, physical_trackman)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context, profiles], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    with np.load(root / "outputs/v17_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v17 OOF rows do not align")

    model = CatBoostClassifier(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="Logloss",
        task_type="GPU", devices="0", random_seed=790 + args.valid_year,
        allow_writing_files=False, verbose=0,
    )
    model.fit(features.loc[train], target[train])
    prediction = model.predict_proba(features.loc[valid])[:, 1]
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for weight in np.arange(0., .451, .025):
        blended = base.copy()
        combined = sigmoid(
            (1. - weight) * logit(base[regular])
            + weight * logit(prediction[regular])
        )
        blended[regular] = combined
        report = {
            "weight": float(weight), "gain": bss(y, blended) - bss(y, base),
            "gain_first_half": bss(y[:midpoint], blended[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
            "gain_second_half": bss(y[midpoint:], blended[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
            "standalone_bss": bss(y[regular], prediction[regular]),
        }
        report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
        reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / f"research/trackman_extended_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        prediction=prediction.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
        feature_names=np.asarray(features.columns),
    )
    print(json.dumps({
        "year": args.valid_year, "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "profile_features": profiles.shape[1], "top": reports[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
