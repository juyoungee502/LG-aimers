"""Test whether calendar month suppresses the v17 Trackman specialist signal.

The candidate is identical to the deployed numeric-ID Trackman specialist except
that ``game_month`` is removed.  It is evaluated as a paired replacement of the
existing v17 component inside the final v23 predictions.
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


def segment_gains(target, base, candidate, rows):
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }
    return {
        name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
        for name, mask in masks.items() if mask.any()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    context = attach_context(data, trackman)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1)
    features = features.drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train_mask = seasons < args.valid_year
    valid_mask = seasons == args.valid_year
    candidate_specialist = fit_predict(
        features, target, train_mask, valid_mask, SEEDS,
    )

    with np.load(root / "outputs" / "v23_oof_predictions.npz") as loaded:
        v23 = {key: loaded[key] for key in loaded.files}
    mask = v23["season"] == args.valid_year
    y = v23["target"][mask].astype(float)
    base = v23["blended"][mask].astype(float)
    old_specialist = v23["trackman_context"][mask].astype(float)
    if not np.allclose(y, target[valid_mask]):
        raise ValueError("v23 OOF rows do not align")
    rows = data.loc[valid_mask].reset_index(drop=True)
    regular = rows["game_type"].eq("R").to_numpy()
    direction = np.zeros(len(y), dtype=float)
    direction[regular] = BLEND_WEIGHT * (
        logit(candidate_specialist[regular]) - logit(old_specialist[regular])
    )

    reports = []
    for scale in np.arange(-.5, 1.501, .05):
        candidate = sigmoid(logit(base) + scale * direction)
        gains = segment_gains(y, base, candidate, rows)
        reports.append({
            "scale": float(scale), "gains": gains,
            "min_half": min(gains["first_half"], gains["second_half"]),
            "min_month_segment": min(
                gains[name] for name in ("months_3_5", "months_6_7", "months_8_11")
            ),
        })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    output = root / "research" / f"v23_trackman_no_month_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        old_specialist=old_specialist.astype(np.float32),
        no_month_specialist=candidate_specialist.astype(np.float32),
        direction=direction.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "valid_year": args.valid_year, "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "old_specialist_bss": bss(y, old_specialist),
        "no_month_specialist_bss": bss(y, candidate_specialist),
        "exact_replacement": next(row for row in reports if abs(row["scale"] - 1.0) < 1e-8),
        "top": reports[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
