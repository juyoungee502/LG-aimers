"""Evaluate CatBoost specialists trained on matching parts of prior seasons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def phase(month):
    return np.select([month <= 5, month <= 7], [0, 1], default=2).astype(np.int8)


def weights(seasons, reference):
    age = np.maximum(0., reference - seasons.astype(np.float64))
    return np.exp(-np.log(2.) * age / 3.).astype(np.float32)


def bss(y, prediction):
    rate = float(y.mean())
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / (rate * (1. - rate))))


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def blend(base, candidate, weight, mode):
    if mode == "probability":
        return (1. - weight) * base + weight * candidate
    return sigmoid((1. - weight) * logit(base) + weight * logit(candidate))


def parameters(seed):
    return dict(
        iterations=1400, learning_rate=.018, depth=7, loss_function="Logloss",
        eval_metric="Logloss", l2_leaf_reg=200., random_strength=1.5,
        border_count=32, random_seed=seed, task_type="GPU", devices="0",
        allow_writing_files=False, verbose=0,
    )


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    # Explicit recent-form summaries make the specialist less reliant on tree
    # approximations of a weighted average.
    features["recent_success_mean"] = (
        .2 * features["asof_pitcher_prev1_game_success_rate"]
        + .3 * features["asof_pitcher_prev3_game_success_rate"]
        + .5 * features["asof_pitcher_prev5_game_success_rate"]
    ).astype(np.float32)
    features["recent_middle_mean"] = (
        .2 * features["asof_pitcher_prev1_game_middle_rate"]
        + .3 * features["asof_pitcher_prev3_game_middle_rate"]
        + .5 * features["asof_pitcher_prev5_game_middle_rate"]
    ).astype(np.float32)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)
    phases = phase(raw["game_month"].to_numpy(np.int16))
    regular = raw["game_type"].eq("R").to_numpy()
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}

    reports = []
    stored = {}
    for variant in ("numeric", "categorical"):
        fold_predictions = {}
        for year in (2023, 2024):
            year_prediction = np.full((seasons == year).sum(), np.nan, dtype=np.float64)
            year_phase = phases[seasons == year]
            year_regular = regular[seasons == year]
            for phase_value in range(3):
                train_mask = (seasons < year) & (phases == phase_value) & regular
                valid_global = (seasons == year) & (phases == phase_value) & regular
                valid_local = (year_phase == phase_value) & year_regular
                model = CatBoostClassifier(**parameters(4100 + phase_value + (10 if variant == "categorical" else 0)))
                kwargs = {"cat_features": CAT_COLUMNS} if variant == "categorical" else {}
                model.fit(
                    features.loc[train_mask], target[train_mask],
                    sample_weight=weights(seasons[train_mask], year - 1), **kwargs,
                )
                year_prediction[valid_local] = model.predict_proba(features.loc[valid_global])[:, 1]
                print(f"Complete {variant} year={year} phase={phase_value}", flush=True)
            fold_predictions[year] = year_prediction
        stored[variant] = fold_predictions

        for mode in ("probability", "logit"):
            for weight in np.arange(0., .501, .025):
                gains = {}
                full_gains = {}
                for year in (2023, 2024):
                    fold = oof["season"] == year
                    y = oof["target"][fold].astype(float)
                    base = oof["blended"][fold].astype(float)
                    prediction = base.copy()
                    row_phase = phases[seasons == year]
                    row_regular = regular[seasons == year]
                    for phase_value in range(3):
                        mask = (row_phase == phase_value) & row_regular
                        prediction[mask] = blend(
                            base[mask], fold_predictions[year][mask], weight, mode,
                        )
                        gains[f"{year}_phase{phase_value}"] = (
                            bss(y[mask], prediction[mask]) - bss(y[mask], base[mask])
                        )
                    full_gains[str(year)] = bss(y, prediction) - bss(y, base)
                reports.append({"variant": variant, "mode": mode, "weight": float(weight),
                                "gains": gains, "full_gains": full_gains,
                                "min_phase": min(gains.values()),
                                "mean_phase": float(np.mean(list(gains.values()))),
                                "min_full": min(full_gains.values())})
    reports.sort(key=lambda row: (row["min_phase"], row["min_full"], row["mean_phase"]), reverse=True)
    output = root / "research/season_phase_specialist_v19.npz"
    np.savez_compressed(
        output, reports_json=np.asarray(json.dumps(reports)),
        **{f"{variant}_{year}": prediction.astype(np.float32)
           for variant, folds in stored.items() for year, prediction in folds.items()},
    )
    print(json.dumps(reports[:60], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
