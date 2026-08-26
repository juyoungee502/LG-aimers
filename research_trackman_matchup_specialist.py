"""Measure the incremental value of prior-season Trackman matchup features."""
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
from trackman_matchup import (
    FEATURE_COLUMNS, attach_matchup_context, batter_mapping, prepare_matchups,
)


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def model(seed: int):
    return CatBoostClassifier(
        iterations=1400, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="Logloss",
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
    columns = [
        "trackman_id", "season", "pitcher_trackman_id", "batter_trackman_id",
        "pitcher_hand", "batter_hand", "pitch_type_group", "balls_before",
        "strikes_before", "rel_speed",
    ]
    raw_trackman = pd.read_csv(
        root / "data/trackman_history.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    pitcher_map, pitcher_report = pitcher_mapping(root, data, raw_trackman)
    batter_map, batter_report = batter_mapping(root, data, raw_trackman)
    old_context = attach_context(data, prepare_trackman(raw_trackman, pitcher_map))
    matchups = prepare_matchups(raw_trackman, pitcher_map, batter_map)
    new_context = attach_matchup_context(data, matchups)

    bases = training_history_arrays(data, target_series)
    core = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(core, data)
    core = add_state_interactions(core)
    reference_features = pd.concat([core, old_context], axis=1)
    enriched_features = pd.concat([reference_features, new_context], axis=1)
    for column in CAT_COLUMNS:
        reference_features[column] = reference_features[column].fillna(-1).astype(np.int32)
        enriched_features[column] = enriched_features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    seed = 941 + args.valid_year
    reference_model = model(seed)
    reference_model.fit(reference_features.loc[train], target[train])
    reference = reference_model.predict_proba(reference_features.loc[valid])[:, 1]
    print(f"Reference model complete: {args.valid_year}", flush=True)
    enriched_model = model(seed)
    enriched_model.fit(enriched_features.loc[train], target[train])
    enriched = enriched_model.predict_proba(enriched_features.loc[valid])[:, 1]
    print(f"Enriched model complete: {args.valid_year}", flush=True)

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    delta = logit(enriched) - logit(reference)
    midpoint = len(y) // 2
    reports = []
    for weight in np.arange(-.5, 1.001, .05):
        blended = base.copy()
        blended[regular] = sigmoid(logit(base[regular]) + weight * delta[regular])
        report = {
            "weight": float(weight),
            "gain": bss(y, blended) - bss(y, base),
            "gain_first_half": bss(y[:midpoint], blended[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
            "gain_second_half": bss(y[midpoint:], blended[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
            "delta_std": float(delta[regular].std()),
            "reference_bss_R": bss(y[regular], reference[regular]),
            "enriched_bss_R": bss(y[regular], enriched[regular]),
        }
        report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
        reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / f"research/trackman_matchup_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        reference=reference.astype(np.float32), enriched=enriched.astype(np.float32),
        delta=delta.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
        feature_names=np.asarray(FEATURE_COLUMNS),
    )
    print(json.dumps({
        "year": args.valid_year,
        "mapped_pitchers": len(pitcher_map),
        "mapped_batters": len(batter_map),
        "pitcher_confidence_min": float(pitcher_report["confidence"].min()),
        "batter_confidence_min": float(batter_report["confidence"].min()),
        "matched_trackman_rows": len(matchups),
        "feature_coverage": {
            column: float(new_context.loc[valid, column].notna().mean())
            for column in FEATURE_COLUMNS
        },
        "top": reports[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
