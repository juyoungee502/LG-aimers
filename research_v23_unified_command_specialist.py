"""Validate one deployable no-month Trackman + command specialist over v23.

All command statistics are built from seasons strictly before the queried row.
The model is evaluated as a paired replacement of v23's Trackman specialist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_joint_command_history import add_pitcher_season_exposure
from research_v23_prior_command_context import prior_command_features
from trackman_context import attach_context, pitcher_mapping, prepare_trackman
from train_trackman_context_specialist import BLEND_WEIGHT, SEEDS, fit_predict


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def segment_masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    trackman_features = attach_context(data, trackman)
    full_command = prior_command_features(data, target)
    recent_command = prior_command_features(data, target, history_window=1).rename(
        columns=lambda name: f"recent_{name}"
    )
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat(
        [features, trackman_features, full_command, recent_command], axis=1,
    ).drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train_mask = seasons < args.valid_year
    valid_mask = seasons == args.valid_year
    prediction = fit_predict(features, target, train_mask, valid_mask, SEEDS)

    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        v23 = {key: loaded[key] for key in loaded.files}
    fold = v23["season"] == args.valid_year
    y = v23["target"][fold].astype(float)
    base = v23["blended"][fold].astype(float)
    old = v23["trackman_context"][fold].astype(float)
    if not np.allclose(y, target[valid_mask]):
        raise ValueError("v23 OOF rows do not align")
    exposure_rows = add_pitcher_season_exposure(data)
    rows = exposure_rows.loc[valid_mask].reset_index(drop=True)
    regular = rows["game_type"].eq("R").to_numpy()
    exposure = rows["pitcher_season_n"].to_numpy(float)
    direction = np.zeros(len(y), dtype=float)
    direction[regular] = BLEND_WEIGHT * (
        logit(prediction[regular]) - logit(old[regular])
    )
    gates = {
        "all": np.ones(len(y)),
        "pitch_n_400": (exposure <= 400).astype(float),
        "pitch_n_600": (exposure <= 600).astype(float),
        "pitch_n_800": (exposure <= 800).astype(float),
        "pitch_decay_300": 300.0 / (300.0 + exposure),
        "pitch_decay_600": 600.0 / (600.0 + exposure),
    }
    masks = segment_masks(rows)
    reports = []
    for gate_name, gate in gates.items():
        for scale in np.arange(-.50, 1.501, .025):
            candidate = sigmoid(logit(base) + scale * gate * direction)
            gains = {
                name: bss(y[mask], candidate[mask]) - bss(y[mask], base[mask])
                for name, mask in masks.items() if mask.any()
            }
            reports.append({
                "gate": gate_name, "scale": float(scale), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_month": min(
                    gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                ),
            })
    reports.sort(
        key=lambda row: (
            min(row["min_half"], row["min_month"]), row["gains"]["all"],
        ),
        reverse=True,
    )
    output = root / "research" / (
        f"v23_unified_command_specialist_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        old_specialist=old.astype(np.float32),
        unified_specialist=prediction.astype(np.float32),
        direction=direction.astype(np.float32),
        pitcher_season_n=exposure.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    exact = [
        row for row in reports
        if row["gate"] == "all" and abs(row["scale"] - 1.0) < 1e-8
    ][0]
    print(json.dumps({
        "valid_year": args.valid_year,
        "feature_count": features.shape[1],
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "standalone_bss": bss(y, prediction),
        "exact_replacement": exact,
        "top": reports[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
