"""Validate prior-season pitcher command context inside the Trackman specialist."""
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
CONTEXT_SPECS = (
    ("hand", ("pitcher_id", "batter_hand"), 200.0),
    ("count", ("pitcher_id", "count_state"), 300.0),
    ("hand_count", ("pitcher_id", "batter_hand", "count_state"), 500.0),
)


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def prior_command_features(
    rows: pd.DataFrame, target: np.ndarray, history_window: int | None = None,
) -> pd.DataFrame:
    """Use only seasons before each row's season to build pitcher context rates."""
    work = rows[["season", "pitcher_id", "batter_hand", "balls_before", "strikes_before"]].copy()
    work["count_state"] = (
        work["balls_before"] * 3 + work["strikes_before"]
    ).astype(np.int8)
    work["_target"] = target
    parts = []
    for season in np.sort(work["season"].unique()):
        query = work.loc[work["season"].eq(season)].drop(columns=["_target"]).copy()
        query["_order"] = query.index
        source_mask = work["season"].lt(season)
        if history_window is not None:
            source_mask &= work["season"].ge(season - history_window)
        source = work.loc[source_mask]
        global_rate = float(source["_target"].mean()) if len(source) else .5
        pitcher = source.groupby("pitcher_id", observed=True)["_target"].agg(
            pitcher_sum="sum", pitcher_n="count",
        ).reset_index()
        pitcher["prior_command_pitcher_rate"] = (
            pitcher["pitcher_sum"] + 500.0 * global_rate
        ) / (pitcher["pitcher_n"] + 500.0)
        query = query.merge(
            pitcher[["pitcher_id", "pitcher_n", "prior_command_pitcher_rate"]],
            on="pitcher_id", how="left", sort=False,
        )
        query["pitcher_n"] = query["pitcher_n"].fillna(0.0)
        query["prior_command_pitcher_rate"] = query[
            "prior_command_pitcher_rate"
        ].fillna(global_rate)
        for name, keys, shrink in CONTEXT_SPECS:
            table = source.groupby(list(keys), observed=True)["_target"].agg(
                context_sum="sum", context_n="count",
            ).reset_index()
            table = table.merge(
                pitcher[["pitcher_id", "prior_command_pitcher_rate"]],
                on="pitcher_id", how="left",
            )
            table[f"prior_command_{name}_rate"] = (
                table["context_sum"]
                + shrink * table["prior_command_pitcher_rate"].fillna(global_rate)
            ) / (table["context_n"] + shrink)
            table[f"prior_command_{name}_weight"] = table["context_n"] / (
                table["context_n"] + shrink
            )
            table = table[[
                *keys, f"prior_command_{name}_rate",
                f"prior_command_{name}_weight",
            ]]
            query = query.merge(table, on=list(keys), how="left", sort=False)
            rate_column = f"prior_command_{name}_rate"
            weight_column = f"prior_command_{name}_weight"
            query[rate_column] = query[rate_column].fillna(
                query["prior_command_pitcher_rate"]
            )
            query[weight_column] = query[weight_column].fillna(0.0)
            query[f"prior_command_{name}_delta"] = (
                query[rate_column] - query["prior_command_pitcher_rate"]
            )
        columns = [
            "_order", "prior_command_pitcher_rate",
            *[
                f"prior_command_{name}_{suffix}"
                for name, _, _ in CONTEXT_SPECS
                for suffix in ("rate", "weight", "delta")
            ],
        ]
        parts.append(query[columns])
    joined = pd.concat(parts).sort_values("_order").drop(columns="_order")
    joined.index = rows.index
    return joined.astype(np.float32)


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
    parser.add_argument("--history-window", type=int, default=0)
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
    mapping, _ = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    trackman_features = attach_context(data, trackman)
    history_window = args.history_window or None
    command_features = prior_command_features(data, target, history_window)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, trackman_features, command_features], axis=1)
    features = features.drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    train_mask = seasons < args.valid_year
    valid_mask = seasons == args.valid_year
    prediction = fit_predict(features, target, train_mask, valid_mask, SEEDS)
    with np.load(root / "outputs" / "v23_oof_predictions.npz") as loaded:
        v23 = {key: loaded[key] for key in loaded.files}
    with np.load(
        root / "research" / f"v23_trackman_no_month_{args.valid_year}.npz"
    ) as loaded:
        no_month = loaded["no_month_specialist"].astype(float)
    fold = v23["season"] == args.valid_year
    y = v23["target"][fold].astype(float)
    base = v23["blended"][fold].astype(float)
    old = v23["trackman_context"][fold].astype(float)
    if not np.allclose(y, target[valid_mask]):
        raise ValueError("v23 OOF rows do not align")
    rows = data.loc[valid_mask].reset_index(drop=True)
    regular = rows["game_type"].eq("R").to_numpy()
    directions = {}
    for name, reference in (("combined", old), ("command_increment", no_month)):
        direction = np.zeros(len(y), dtype=float)
        direction[regular] = BLEND_WEIGHT * (
            logit(prediction[regular]) - logit(reference[regular])
        )
        directions[name] = direction

    reports = []
    for name, direction in directions.items():
        for scale in np.arange(-.5, 1.501, .05):
            candidate = sigmoid(logit(base) + scale * direction)
            gains = segment_gains(y, base, candidate, rows)
            reports.append({
                "direction": name, "scale": float(scale), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_month_segment": min(
                    gains[key] for key in (
                        "months_3_5", "months_6_7", "months_8_11",
                    )
                ),
                "standalone_bss": bss(y, prediction),
            })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    suffix = f"_w{args.history_window}" if args.history_window else ""
    output = root / "research" / (
        f"v23_prior_command_context_{args.valid_year}{suffix}.npz"
    )
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        prediction=prediction.astype(np.float32),
        combined_direction=directions["combined"].astype(np.float32),
        command_direction=directions["command_increment"].astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    exact = [row for row in reports if abs(row["scale"] - 1.0) < 1e-8]
    print(json.dumps({
        "valid_year": args.valid_year, "history_window": history_window,
        "feature_count": features.shape[1],
        "exact": exact, "top": reports[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
