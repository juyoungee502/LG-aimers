"""Test previous-season Trackman repertoire alongside all-history context."""
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
from trackman_context import (
    FEATURE_COLUMNS, attach_context, deviation_table, pitcher_mapping,
    prepare_trackman, summarize,
)
from train_trackman_context_specialist import BLEND_WEIGHT, SEEDS, fit_predict


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
RECENT_COLUMNS = tuple(f"recent_{column}" for column in FEATURE_COLUMNS)


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def recent_tables(trackman, target_season):
    past = trackman.loc[trackman["season"].eq(target_season - 1)]
    if past.empty:
        return None, None
    baseline = summarize(past, ["pitcher_id"], "base", 0)
    count = deviation_table(
        summarize(
            past, ["pitcher_id", "balls_before", "strikes_before"],
            "trackman_count", 15,
        ), baseline, "trackman_count",
    ).add_prefix("recent_")
    hand = deviation_table(
        summarize(
            past, ["pitcher_id", "batter_hand"], "trackman_hand", 25,
        ), baseline, "trackman_hand",
    ).add_prefix("recent_")
    return count, hand


def attach_recent(rows, trackman):
    blocks = []
    hand_codes = {1: "Left", 2: "Right"}
    count_columns = [column for column in RECENT_COLUMNS if "_count_" in column]
    hand_columns = [column for column in RECENT_COLUMNS if "_hand_" in column]
    for season, part in rows.groupby("season", sort=False):
        count, hand = recent_tables(trackman, int(season))
        output = pd.DataFrame(index=part.index, columns=RECENT_COLUMNS, dtype=np.float32)
        if count is None:
            blocks.append(output)
            continue
        left = part[["pitcher_id", "balls_before", "strikes_before", "batter_hand"]].copy()
        left["_order"] = np.arange(len(left))
        left["batter_hand_name"] = left["batter_hand"].map(hand_codes)
        got_count = left.merge(
            count.reset_index(),
            on=["pitcher_id", "balls_before", "strikes_before"],
            how="left", sort=False,
        ).sort_values("_order")
        got_hand = left.merge(
            hand.reset_index().rename(columns={"batter_hand": "batter_hand_name"}),
            on=["pitcher_id", "batter_hand_name"], how="left", sort=False,
        ).sort_values("_order")
        output[count_columns] = got_count[count_columns].to_numpy(np.float32)
        output[hand_columns] = got_hand[hand_columns].to_numpy(np.float32)
        blocks.append(output)
    return pd.concat(blocks).sort_index().reindex(columns=RECENT_COLUMNS)


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
        for name, mask in masks.items()
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
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    context = attach_context(data, trackman)
    recent = attach_recent(data, trackman)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context, recent], axis=1).drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    enriched = fit_predict(features, target, train, valid, SEEDS)

    with np.load(root / "outputs/v23_oof_predictions.npz") as z:
        fold = z["season"] == args.valid_year
        y = z["target"][fold].astype(float)
        base = z["blended"][fold].astype(float)
    with np.load(root / f"research/v23_trackman_no_month_{args.valid_year}.npz") as z:
        reference = z["no_month_specialist"].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v23 OOF rows do not align")
    rows = data.loc[valid].reset_index(drop=True)
    regular = rows["game_type"].eq("R").to_numpy()
    direction = np.zeros(len(y), dtype=float)
    direction[regular] = BLEND_WEIGHT * (
        logit(enriched[regular]) - logit(reference[regular])
    )
    reports = []
    for weight in np.arange(-.75, 1.501, .05):
        candidate = sigmoid(logit(base) + weight * direction)
        gains = segment_gains(y, base, candidate, rows)
        reports.append({
            "weight": float(weight), "gains": gains,
            "min_temporal": min(
                gains["first_half"], gains["second_half"], gains["months_3_5"],
                gains["months_6_7"], gains["months_8_11"],
            ),
        })
    reports.sort(key=lambda row: (row["min_temporal"], row["gains"]["all"]), reverse=True)
    output = root / f"research/v23_trackman_recent_context_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        reference=reference.astype(np.float32), enriched=enriched.astype(np.float32),
        direction=direction.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "valid_year": args.valid_year, "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "recent_features": len(RECENT_COLUMNS), "top": reports[:50],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
