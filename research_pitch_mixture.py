"""Marginalize a type-conditioned command model over predicted pitch choice."""
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
from research_inferred_pitch_priors import PITCH_TYPES, bss, reconstruct_labels


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    pitch_code = labels["pitch_type"].map(
        {name: index for index, name in enumerate(PITCH_TYPES)}
    )
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    train = (seasons < args.valid_year) & pitch_code.notna().to_numpy()
    valid = seasons == args.valid_year

    selection = CatBoostClassifier(
        iterations=1000, learning_rate=.025, depth=6, l2_leaf_reg=100.0,
        random_strength=1.0, border_count=32, loss_function="MultiClass",
        eval_metric="MultiClass", task_type="GPU", devices="0", random_seed=720,
        allow_writing_files=False, verbose=0,
    )
    selection.fit(
        features.loc[train], pitch_code.loc[train].to_numpy(np.int8),
        cat_features=CAT_COLUMNS,
    )
    learned_selection = selection.predict_proba(features.loc[valid])
    aligned_selection = np.zeros((valid.sum(), len(PITCH_TYPES)), dtype=np.float64)
    for column, label in enumerate(selection.classes_.astype(int)):
        aligned_selection[:, label] = learned_selection[:, column]

    conditional_train = features.loc[train].copy()
    conditional_train["current_pitch_type"] = pitch_code.loc[train].to_numpy(np.int8)
    outcome = CatBoostClassifier(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="Logloss",
        task_type="GPU", devices="0", random_seed=721,
        allow_writing_files=False, verbose=0,
    )
    outcome.fit(
        conditional_train, target[train],
        cat_features=[*CAT_COLUMNS, "current_pitch_type"],
    )
    conditional_probabilities = []
    for pitch_type in range(len(PITCH_TYPES)):
        query = features.loc[valid].copy()
        query["current_pitch_type"] = pitch_type
        conditional_probabilities.append(outcome.predict_proba(query)[:, 1])
    conditional_probabilities = np.column_stack(conditional_probabilities)

    asof_selection = data.loc[valid, [
        f"asof_pitcher_{name}_rate" for name in PITCH_TYPES
    ]].to_numpy(float)
    global_selection = np.bincount(
        pitch_code.loc[train].to_numpy(np.int8), minlength=len(PITCH_TYPES),
    ).astype(float)
    global_selection /= global_selection.sum()
    missing = ~np.isfinite(asof_selection)
    asof_selection[missing] = np.broadcast_to(
        global_selection, asof_selection.shape
    )[missing]
    totals = asof_selection.sum(axis=1, keepdims=True)
    asof_selection = np.divide(
        asof_selection, totals,
        out=np.broadcast_to(global_selection, asof_selection.shape).copy(),
        where=totals > 0,
    )
    with np.load(root / "outputs" / "v16_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    oof_mask = oof["season"] == args.valid_year
    y = oof["target"][oof_mask].astype(float)
    base = oof["blended"][oof_mask].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("OOF rows do not align")
    reports, mixture_predictions = [], {}
    half = len(y) // 2
    for selection_weight in np.arange(0.0, 1.001, .25):
        probabilities = (
            selection_weight * aligned_selection
            + (1.0 - selection_weight) * asof_selection
        )
        mixture = np.sum(probabilities * conditional_probabilities, axis=1)
        mixture_predictions[str(float(selection_weight))] = mixture.astype(np.float32)
        for weight in np.arange(0.0, .401, .025):
            blended = sigmoid((1.0 - weight) * logit(base) + weight * logit(mixture))
            values = [
                bss(y, blended) - bss(y, base),
                bss(y[:half], blended[:half]) - bss(y[:half], base[:half]),
                bss(y[half:], blended[half:]) - bss(y[half:], base[half:]),
            ]
            reports.append({
                "valid_year": args.valid_year,
                "selection_model_weight": float(selection_weight),
                "mixture_weight": float(weight), "gain": values[0],
                "gain_first_half": values[1], "gain_second_half": values[2],
                "min_half": min(values[1:]), "mixture_bss": bss(y, mixture),
            })
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research" / f"pitch_mixture_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        learned_selection=aligned_selection.astype(np.float32),
        asof_selection=asof_selection.astype(np.float32),
        conditional=conditional_probabilities.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    valid_pitch_code = pitch_code.loc[valid].to_numpy(float)
    known_pitch = np.isfinite(valid_pitch_code)
    print(json.dumps({
        "pitch_label_coverage": float(pitch_code.notna().mean()),
        "selection_accuracy": float(np.mean(
            np.argmax(aligned_selection[known_pitch], axis=1)
            == valid_pitch_code[known_pitch].astype(int)
        )),
        "top": reports[:50],
    }, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
