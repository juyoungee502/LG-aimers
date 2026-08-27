"""Audit exact integer representations of cumulative as-of rates over v24."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from exact_asof_features import exact_asof_features
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_combined_candidate import logit, sigmoid
from trackman_context import attach_context, pitcher_mapping, prepare_trackman
from train_trackman_context_specialist import BLEND_WEIGHT, SEEDS, fit_predict


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
warnings.filterwarnings("ignore", category=PerformanceWarning)


def segment_gains(target, base, candidate, rows):
    position = np.arange(len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
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
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, _ = pitcher_mapping(root, data, trackman)
    context = attach_context(data, prepare_trackman(trackman, mapping))
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([
        features.reset_index(drop=True), context.reset_index(drop=True),
        exact_asof_features(data).reset_index(drop=True),
    ], axis=1).drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    prediction = fit_predict(features, target, train, valid, SEEDS)
    with np.load(root / f"research/v23_trackman_no_month_{args.valid_year}.npz") as z:
        reference = z["no_month_specialist"].astype(float)
    with np.load(root / "outputs/v24_oof_predictions.npz") as z:
        active = z["season"] == args.valid_year
        y = z["target"][active].astype(float)
        base = z["blended"][active].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v24 OOF rows do not align")
    rows = data.loc[valid].reset_index(drop=True)
    regular = rows["game_type"].eq("R").to_numpy()
    direction = np.zeros(len(y), dtype=float)
    direction[regular] = BLEND_WEIGHT * (
        logit(prediction[regular]) - logit(reference[regular])
    )
    reports = []
    for scale in np.arange(-.5, 1.501, .025):
        candidate = sigmoid(logit(base) + scale * direction)
        gains = segment_gains(y, base, candidate, rows)
        reports.append({
            "scale": float(scale), "gains": gains,
            "min_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
            "min_month": min(gains[name] for name in (
                "months_3_5", "months_6_7", "months_8_11",
            )),
        })
    reports.sort(key=lambda row: (
        min(row["min_quarter"], row["min_month"]), row["gains"]["all"],
    ), reverse=True)
    output = root / f"research/v24_exact_asof_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        reference=reference.astype(np.float32), prediction=prediction.astype(np.float32),
        direction=direction.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
        feature_names=np.asarray(features.columns),
    )
    exact = min(reports, key=lambda row: abs(row["scale"] - 1.0))
    print(json.dumps({
        "year": args.valid_year,
        "reference_bss_R": bss(y[regular], reference[regular]),
        "exact_asof_bss_R": bss(y[regular], prediction[regular]),
        "delta_logit_sd_R": float(direction[regular].std()),
        "exact_replacement": exact, "top": reports[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
