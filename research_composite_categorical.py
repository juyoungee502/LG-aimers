"""Screen ordered target statistics for explicit entity-context categories."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


BASE_CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
COMPOSITE_COLUMNS = [
    "pitcher_count_cat", "pitcher_batter_hand_cat", "pitcher_game_type_cat",
    "batter_count_cat", "batter_pitcher_hand_cat",
    "pitcher_count_batter_hand_cat",
]
CAT_COLUMNS = [*BASE_CAT_COLUMNS, *COMPOSITE_COLUMNS]
warnings.filterwarnings("ignore", category=PerformanceWarning)


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def add_composites(features, raw):
    count = features["count_state"].to_numpy(np.int64)
    pitcher = raw["pitcher_id"].to_numpy(np.int64)
    batter = raw["batter_id"].to_numpy(np.int64)
    pitcher_hand = raw["pitcher_hand"].to_numpy(np.int64)
    batter_hand = raw["batter_hand"].to_numpy(np.int64)
    game = raw["game_type"].map({"R": 0, "F": 1}).fillna(2).to_numpy(np.int64)
    values = {
        "pitcher_count_cat": pitcher * 16 + count,
        "pitcher_batter_hand_cat": pitcher * 4 + batter_hand,
        "pitcher_game_type_cat": pitcher * 4 + game,
        "batter_count_cat": batter * 16 + count,
        "batter_pitcher_hand_cat": batter * 4 + pitcher_hand,
        "pitcher_count_batter_hand_cat": pitcher * 64 + count * 4 + batter_hand,
    }
    return pd.concat([
        features,
        pd.DataFrame(values, index=features.index).astype(np.int64),
    ], axis=1)


def parameters(seed):
    return dict(
        iterations=1400, learning_rate=.018, depth=7, l2_leaf_reg=150.,
        random_strength=1.25, border_count=32, loss_function="Logloss",
        eval_metric="Logloss", max_ctr_complexity=1, one_hot_max_size=16,
        task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
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
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = add_composites(features, raw)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int64)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < valid_year
    valid = seasons == valid_year
    age = (valid_year - 1) - seasons[train].astype(np.float64)
    sample_weight = np.exp(-np.log(2.) * age / 3.).astype(np.float32)
    model = CatBoostClassifier(**parameters(830 + valid_year))
    model.fit(
        features.loc[train], target[train],
        sample_weight=sample_weight, cat_features=CAT_COLUMNS,
    )
    prediction = model.predict_proba(features.loc[valid])[:, 1]
    print(f"Composite categorical model complete: year={valid_year}", flush=True)

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
        for weight in np.arange(-.2, .601, .01):
            blended = base.copy()
            blended[selected] = sigmoid(
                (1. - weight) * logit(base[selected])
                + weight * logit(prediction[selected])
            )
            values = [
                bss(y, blended) - bss(y, base),
                bss(y[:midpoint], blended[:midpoint])
                - bss(y[:midpoint], base[:midpoint]),
                bss(y[midpoint:], blended[midpoint:])
                - bss(y[midpoint:], base[midpoint:]),
            ]
            reports.append({
                "year": valid_year, "gate": gate, "weight": float(weight),
                "gain": values[0], "gain_first_half": values[1],
                "gain_second_half": values[2], "min_half": min(values[1:]),
                "standalone_bss": bss(y[selected], prediction[selected]),
                "prediction_mean": float(prediction[selected].mean()),
                "target_mean": float(y[selected].mean()),
            })
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research" / f"composite_categorical_{valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        regular=regular, prediction=prediction.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
        feature_names=np.asarray(features.columns),
    )
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
