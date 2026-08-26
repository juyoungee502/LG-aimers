"""Latent pitch-outcome mixture of experts using training-only labels.

Ball/strike/other is reconstructed for training rows, but is never an inference
input.  A pre-pitch gating model predicts that latent outcome and combines
three success experts trained within the corresponding outcome strata.
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


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
OUTCOMES = ("ball", "strike", "other")


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed: int, multiclass: bool = False):
    loss = "MultiClass" if multiclass else "Logloss"
    return dict(
        iterations=1400, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function=loss, eval_metric=loss,
        task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    usable = labels[["ball", "strike"]].notna().all(axis=1).to_numpy()
    outcome = np.full(len(data), -1, dtype=np.int8)
    outcome[usable & labels["ball"].eq(1).to_numpy()] = 0
    outcome[usable & labels["strike"].eq(1).to_numpy()] = 1
    outcome[usable & labels["ball"].eq(0).to_numpy()
            & labels["strike"].eq(0).to_numpy()] = 2
    if not np.array_equal(usable, outcome >= 0):
        raise ValueError("Outcome reconstruction is not mutually exclusive")

    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    train = (seasons < args.valid_year) & usable
    valid = seasons == args.valid_year

    gate = CatBoostClassifier(**parameters(4100 + args.valid_year, multiclass=True))
    gate.fit(features.loc[train], outcome[train])
    gate_probability = gate.predict_proba(features.loc[valid])
    class_positions = [list(gate.classes_).index(index) for index in range(3)]
    gate_probability = gate_probability[:, class_positions]
    print(f"Outcome gate complete: {args.valid_year}", flush=True)

    expert_probability = []
    for index, name in enumerate(OUTCOMES):
        mask = train & (outcome == index)
        expert = CatBoostClassifier(**parameters(4200 + index, multiclass=False))
        expert.fit(features.loc[mask], target[mask])
        expert_probability.append(expert.predict_proba(features.loc[valid])[:, 1])
        print(
            f"Outcome expert complete: year={args.valid_year}, "
            f"outcome={name}, rows={mask.sum()}", flush=True,
        )
    expert_probability = np.column_stack(expert_probability)
    mixture = np.sum(gate_probability * expert_probability, axis=1)

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for weight in np.arange(-.1, .501, .01):
        prediction = base.copy()
        prediction[regular] = sigmoid(
            (1. - weight) * logit(base[regular])
            + weight * logit(mixture[regular])
        )
        report = {
            "weight": float(weight),
            "gain": bss(y, prediction) - bss(y, base),
            "gain_first_half": bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
            "gain_second_half": bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
            "mixture_bss_R": bss(y[regular], mixture[regular]),
            "mixture_mean_R": float(mixture[regular].mean()),
            "target_mean_R": float(y[regular].mean()),
        }
        report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
        reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    valid_usable = usable[valid]
    gate_choice = np.argmax(gate_probability[valid_usable], axis=1)
    output = root / f"research/outcome_moe_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        gate=gate_probability.astype(np.float32),
        experts=expert_probability.astype(np.float32), mixture=mixture.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)), outcomes=np.asarray(OUTCOMES),
    )
    print(json.dumps({
        "year": args.valid_year, "usable_train": int(train.sum()),
        "gate_accuracy": float(np.mean(gate_choice == outcome[valid][valid_usable])),
        "gate_mean": dict(zip(OUTCOMES, map(float, gate_probability.mean(axis=0)))),
        "top": reports[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
