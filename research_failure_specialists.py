"""Screen decomposed failure classifiers as small logit blends over v16."""
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
from research_inferred_pitch_priors import bss, reconstruct_labels


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
LABELS = ("reverse", "middle", "wayoff")


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed, variant):
    if variant == "weighted_depth6":
        return dict(
            iterations=1200, learning_rate=.02, depth=6, l2_leaf_reg=100.,
            random_strength=1., border_count=32,
        )
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    args = parser.parse_args()
    valid_year = args.valid_year
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    labels = reconstruct_labels(pd.concat([raw, target_series.rename(TARGET_COL)], axis=1))
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < valid_year
    valid = seasons == valid_year
    usable = labels[list(LABELS)].notna().all(axis=1).to_numpy() & train

    with np.load(root / "outputs" / "v16_oof_predictions.npz", allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    latest = oof["season"] == valid_year
    y = oof["target"][latest].astype(float)
    base = oof["blended"][latest].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError(f"v16 OOF and train.csv do not align for {valid_year}")

    variants, variant_predictions, reports = (
        ("weighted_depth6", "uniform_depth8"), {}, []
    )
    for variant in variants:
        predictions = {}
        age = (valid_year - 1) - seasons[usable].astype(float)
        sample_weight = (
            np.exp(-np.log(2.) * age / 3.).astype(np.float32)
            if variant == "weighted_depth6" else None
        )
        for offset, label in enumerate(LABELS):
            model = CatBoostClassifier(
                **parameters(250 + offset, variant), loss_function="Logloss",
                eval_metric="Logloss", task_type="GPU", devices="0",
                random_seed=250 + offset, allow_writing_files=False, verbose=0,
            )
            model.fit(
                features.loc[usable], labels.loc[usable, label].to_numpy(np.int8),
                sample_weight=sample_weight,
            )
            predictions[label] = model.predict_proba(features.loc[valid])[:, 1]
        p_all = np.clip(
            1. - predictions["reverse"] - predictions["middle"]
            - predictions["wayoff"], 1e-5, 1. - 1e-5,
        )
        p_middle = np.clip(1. - predictions["middle"], 1e-5, 1. - 1e-5)
        variant_predictions[variant] = np.column_stack([
            predictions[name] for name in LABELS
        ] + [p_all, p_middle])
        halfway = len(y) // 2
        for weight_all in np.arange(0., .151, .01):
            for weight_middle in np.arange(0., .051, .005):
                prediction = sigmoid(
                    (1. - weight_all - weight_middle) * logit(base)
                    + weight_all * logit(p_all) + weight_middle * logit(p_middle)
                )
                gains = [
                    bss(y, prediction) - bss(y, base),
                    bss(y[:halfway], prediction[:halfway])
                    - bss(y[:halfway], base[:halfway]),
                    bss(y[halfway:], prediction[halfway:])
                    - bss(y[halfway:], base[halfway:]),
                ]
                reports.append({
                    "variant": variant, "weight_all": float(weight_all),
                    "weight_middle": float(weight_middle), "gain_2024": gains[0],
                    "gain_first_half": gains[1], "gain_second_half": gains[2],
                    "min_half": min(gains[1:]),
                    "valid_year": valid_year, "p_all_bss": bss(y, p_all),
                    "p_middle_bss": bss(y, p_middle),
                })
    reports.sort(key=lambda row: (row["min_half"], row["gain_2024"]), reverse=True)
    output = root / "research" / f"failure_specialists_{valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, variants=np.asarray(variants),
        predictions=np.stack([variant_predictions[name] for name in variants]),
        target=y.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps(reports[:40], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
