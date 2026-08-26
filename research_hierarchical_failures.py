"""Evaluate an exact conditional factorisation of inferred pitch failures.

Success is the event that reverse, middle, and wayoff are all false.  Reverse
and middle overlap, so subtracting three marginal probabilities (the v19
specialist) double counts that intersection.  This experiment learns the
required conditional probabilities instead of estimating the intersection.
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


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed):
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="Logloss",
        task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def fit_predict(features, labels, train_rows, valid_rows, label, seed):
    model = CatBoostClassifier(**parameters(seed))
    model.fit(
        features.loc[train_rows], labels.loc[train_rows, label].to_numpy(np.int8),
    )
    return model.predict_proba(features.loc[valid_rows])[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    args = parser.parse_args()
    valid_year = args.valid_year
    root = Path(__file__).resolve().parent

    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    labels = reconstruct_labels(
        pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    )
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(
        raw, *bases, global_prior=float(target_series.mean())
    )
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([
        features,
        prior_season_context(
            pd.concat([raw, target_series.rename(TARGET_COL)], axis=1), labels,
        ),
    ], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < valid_year
    valid = seasons == valid_year
    complete = labels[["reverse", "middle", "wayoff"]].notna().all(axis=1).to_numpy()
    train_complete = train & complete

    reverse = labels["reverse"].fillna(0).to_numpy(np.int8)
    middle = labels["middle"].fillna(0).to_numpy(np.int8)
    train_no_reverse = train_complete & (reverse == 0)
    train_no_middle = train_complete & (middle == 0)
    train_neither = train_complete & (reverse == 0) & (middle == 0)

    # Reuse the already-screened unconditional specialists.  Only the three
    # genuinely conditional distributions need new models.
    with np.load(
        root / f"research/failure_specialists_{valid_year}_prior_context.npz"
    ) as loaded:
        stored = {key: loaded[key] for key in loaded.files}
    variant_index = list(stored["variants"].astype(str)).index("uniform_depth8")
    unconditional = stored["predictions"][variant_index, :, :3].astype(np.float64)
    p_reverse, p_middle = unconditional[:, 0], unconditional[:, 1]

    p_middle_no_reverse = fit_predict(
        features, labels, train_no_reverse, valid,
        "middle", 611,
    )
    print(f"Conditional model complete: year={valid_year}, middle|no_reverse", flush=True)
    p_reverse_no_middle = fit_predict(
        features, labels, train_no_middle, valid,
        "reverse", 612,
    )
    print(f"Conditional model complete: year={valid_year}, reverse|no_middle", flush=True)
    p_wayoff_neither = fit_predict(
        features, labels, train_neither, valid,
        "wayoff", 613,
    )
    print(f"Conditional model complete: year={valid_year}, wayoff|neither", flush=True)

    candidates = {
        "reverse_first": (
            (1. - p_reverse) * (1. - p_middle_no_reverse)
            * (1. - p_wayoff_neither)
        ),
        "middle_first": (
            (1. - p_middle) * (1. - p_reverse_no_middle)
            * (1. - p_wayoff_neither)
        ),
    }
    candidates["order_average"] = .5 * (
        candidates["reverse_first"] + candidates["middle_first"]
    )

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    oof_rows = oof["season"] == valid_year
    y = oof["target"][oof_rows].astype(np.float64)
    base = oof["blended"][oof_rows].astype(np.float64)
    if not np.allclose(y, stored["target"]):
        raise ValueError(f"Failure OOF rows differ for {valid_year}")
    regular = raw.loc[valid, "game_type"].eq("R").to_numpy()

    reports = []
    midpoint = len(y) // 2
    for name, candidate in candidates.items():
        candidate = np.clip(candidate, 1e-5, 1. - 1e-5)
        for gate, selected in (("R", regular), ("all", np.ones(len(y), dtype=bool))):
            for weight in np.arange(-.1, .501, .01):
                prediction = base.copy()
                prediction[selected] = sigmoid(
                    (1. - weight) * logit(base[selected])
                    + weight * logit(candidate[selected])
                )
                gains = [
                    bss(y, prediction) - bss(y, base),
                    bss(y[:midpoint], prediction[:midpoint])
                    - bss(y[:midpoint], base[:midpoint]),
                    bss(y[midpoint:], prediction[midpoint:])
                    - bss(y[midpoint:], base[midpoint:]),
                ]
                reports.append({
                    "year": valid_year, "candidate": name, "gate": gate,
                    "weight": float(weight), "gain": gains[0],
                    "gain_first_half": gains[1], "gain_second_half": gains[2],
                    "min_half": min(gains[1:]),
                    "standalone_bss": bss(y[selected], candidate[selected]),
                    "prediction_mean": float(candidate[selected].mean()),
                    "target_mean": float(y[selected].mean()),
                })
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research" / f"hierarchical_failures_{valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        regular=regular,
        candidate_names=np.asarray(list(candidates)),
        candidates=np.stack(list(candidates.values())).astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
