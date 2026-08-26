"""Evaluate one historical-Trackman CatBoost expert per exact pitch count."""
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
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed):
    return dict(
        iterations=1200, learning_rate=.02, depth=6, l2_leaf_reg=100.,
        random_strength=1., border_count=32, loss_function="Logloss",
        eval_metric="Logloss", task_type="GPU", devices="0",
        random_seed=seed, allow_writing_files=False, verbose=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    valid_year = args.valid_year
    root = Path(__file__).resolve().parent

    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    mapping, _ = pitcher_mapping(root, raw, trackman)
    trackman = prepare_trackman(trackman, mapping)
    context = attach_context(raw, trackman)

    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < valid_year
    valid = seasons == valid_year
    count_state = (
        raw["balls_before"].to_numpy(np.int8) * 3
        + raw["strikes_before"].to_numpy(np.int8)
    )
    prediction = np.full(int(valid.sum()), np.nan, dtype=np.float64)
    valid_counts = count_state[valid]
    for count in range(12):
        train_rows = train & (count_state == count)
        valid_local = valid_counts == count
        if not valid_local.any():
            continue
        model = CatBoostClassifier(**parameters(720 + count))
        model.fit(features.loc[train_rows], target[train_rows])
        prediction[valid_local] = model.predict_proba(
            features.loc[valid].loc[valid_local]
        )[:, 1]
        print(
            f"Exact-count expert complete: year={valid_year}, count={count}, "
            f"train={int(train_rows.sum())}, valid={int(valid_local.sum())}",
            flush=True,
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Exact-count specialist left missing predictions")

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError(f"v19 OOF and train.csv differ for {valid_year}")
    regular = raw.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for gate, selected in (("R", regular), ("all", np.ones(len(y), dtype=bool))):
        for weight in np.arange(-.2, .701, .01):
            blended = base.copy()
            blended[selected] = sigmoid(
                (1. - weight) * logit(base[selected])
                + weight * logit(prediction[selected])
            )
            gains = [
                bss(y, blended) - bss(y, base),
                bss(y[:midpoint], blended[:midpoint])
                - bss(y[:midpoint], base[:midpoint]),
                bss(y[midpoint:], blended[midpoint:])
                - bss(y[midpoint:], base[midpoint:]),
            ]
            reports.append({
                "year": valid_year, "gate": gate, "weight": float(weight),
                "gain": gains[0], "gain_first_half": gains[1],
                "gain_second_half": gains[2], "min_half": min(gains[1:]),
                "standalone_bss": bss(y[selected], prediction[selected]),
                "prediction_mean": float(prediction[selected].mean()),
                "target_mean": float(y[selected].mean()),
            })
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research" / f"exact_count_specialist_{valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        regular=regular, prediction=prediction.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
