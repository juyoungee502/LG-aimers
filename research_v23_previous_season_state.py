"""Test a compact, time-safe previous-season state specialist.

The official cumulative as-of columns let us recover each player's activity in
the immediately preceding season.  This model intentionally excludes player
IDs, season, month, and Trackman data so that it learns transferable effects of
last-season command/pitch-mix state instead of memorising identities or time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import TARGET_COL
from research_inferred_pitch_priors import bss


SEEDS = (4701, 4702, 4703)
COMPONENTS = (
    ("pitcher_success", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", True),
    ("pitcher_reverse", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", False),
    ("pitcher_middle", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", False),
    ("pitcher_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", False),
    ("pitcher_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", False),
    ("pitcher_fastball", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", False),
    ("pitcher_breaking", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_breaking_rate", False),
    ("pitcher_offspeed", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_offspeed_rate", False),
    ("batter_success", "batter_id", "asof_batter_n", "asof_batter_success_rate", True),
    ("batter_middle", "batter_id", "asof_batter_n", "asof_batter_middle_rate", False),
)
CAT_COLUMNS = ("game_type", "count_state", "pitcher_hand", "batter_hand")


def previous_season_features(raw: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    """Map exact season Y-1 summaries to every row in season Y."""
    result = pd.DataFrame(index=raw.index)
    seasons = raw["season"].to_numpy(np.int16)
    for name, *_ in COMPONENTS:
        result[f"prev1_{name}_rate"] = np.nan
        result[f"prev1_{name}_s50"] = np.nan
        result[f"prev1_{name}_s200"] = np.nan
        result[f"prev1_{name}_log_n"] = 0.0
    for season in np.sort(raw["season"].unique()):
        query_positions = np.flatnonzero(seasons == season)
        source_positions = np.flatnonzero(seasons == season - 1)
        if not len(source_positions):
            continue
        source = raw.iloc[source_positions]
        query = raw.iloc[query_positions]
        for name, id_col, n_col, rate_col, is_success in COMPONENTS:
            first_positions = source.groupby(id_col, sort=False, observed=True).head(1).index
            last_positions = source.groupby(id_col, sort=False, observed=True).tail(1).index
            first = raw.loc[first_positions]
            last = raw.loc[last_positions]
            start_n = first[n_col].fillna(0.0).to_numpy(float)
            start_count = np.rint(
                start_n * first[rate_col].fillna(0.0).to_numpy(float)
            )
            end_before_n = last[n_col].fillna(0.0).to_numpy(float)
            end_before_rate = last[rate_col].fillna(0.0).to_numpy(float)
            if is_success:
                end_count = np.rint(end_before_n * end_before_rate) + target[last_positions]
            else:
                # The final component label is hidden. Its pre-event rate is an
                # unbiased fractional estimate with at most one-event error.
                end_count = np.rint(end_before_n * end_before_rate) + end_before_rate
            season_n = np.maximum(0.0, end_before_n + 1.0 - start_n)
            season_count = np.clip(end_count - start_count, 0.0, season_n)
            global_rate = float(season_count.sum() / max(season_n.sum(), 1.0))
            table = pd.DataFrame({
                id_col: last[id_col].astype(np.int64).to_numpy(),
                "_n": season_n,
                "_count": season_count,
            })
            ids = query[id_col]
            n_map = dict(zip(table[id_col], table["_n"]))
            count_map = dict(zip(table[id_col], table["_count"]))
            n = ids.map(n_map).fillna(0.0).to_numpy(float)
            count = ids.map(count_map).fillna(0.0).to_numpy(float)
            rate = np.divide(
                count, n, out=np.full(len(query), global_rate), where=n > 0,
            )
            result.loc[query.index, f"prev1_{name}_rate"] = rate
            result.loc[query.index, f"prev1_{name}_s50"] = (
                count + 50.0 * global_rate
            ) / (n + 50.0)
            result.loc[query.index, f"prev1_{name}_s200"] = (
                count + 200.0 * global_rate
            ) / (n + 200.0)
            result.loc[query.index, f"prev1_{name}_log_n"] = np.log1p(n)
    return result.astype(np.float32)


def model_features(raw: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=raw.index)
    features["game_type"] = raw["game_type"].map({"R": 0, "F": 1}).fillna(-1)
    features["count_state"] = raw["balls_before"] * 3 + raw["strikes_before"]
    features["pitcher_hand"] = raw["pitcher_hand"]
    features["batter_hand"] = raw["batter_hand"]
    for column in (
        "num_runners_on", "outs_before", "inning", "li", "score_diff_pitcher_team",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ):
        features[column] = raw[column]
    features["runner_gate"] = raw["num_runners_on"].gt(0).astype(np.int8)
    features["two_strike"] = raw["strikes_before"].eq(2).astype(np.int8)
    features["pitcher_advantage"] = raw["strikes_before"].gt(raw["balls_before"]).astype(np.int8)
    return pd.concat([features, previous], axis=1)


def frozen_prediction_center(prediction, raw, train, valid):
    context = pd.DataFrame({
        "game_type": raw["game_type"].astype(str),
        "count_state": raw["balls_before"] * 3 + raw["strikes_before"],
    })
    source = context.loc[train].copy()
    source["prediction"] = prediction[train]
    table = source.groupby(
        ["game_type", "count_state"], observed=True,
    )["prediction"].mean().reset_index()
    fallback = source.groupby("game_type", observed=True)["prediction"].mean().to_dict()
    query = context.loc[valid].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(
        table, on=["game_type", "count_state"], how="left", sort=False,
    ).sort_values("_order")
    query["prediction"] = query["prediction"].fillna(query["game_type"].map(fallback))
    return query["prediction"].fillna(float(source["prediction"].mean())).to_numpy(float)


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
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target = raw.pop(TARGET_COL).to_numpy(np.float32)
    previous = previous_season_features(raw, target)
    features = model_features(raw, previous)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    label_context = pd.DataFrame({
        "season": seasons,
        "game_type": raw["game_type"].astype(str),
        "count_state": raw["balls_before"] * 3 + raw["strikes_before"],
        "target": target,
    })
    center = label_context.groupby(
        ["season", "game_type", "count_state"], observed=True,
    )["target"].transform("mean").to_numpy(float)
    label = target - center
    valid_members = []
    train_members = []
    cat_indices = [features.columns.get_loc(column) for column in CAT_COLUMNS]
    for seed in SEEDS:
        model = CatBoostRegressor(
            iterations=1200, depth=6, learning_rate=.025,
            loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=200.,
            random_strength=1., border_count=32, bootstrap_type="Bayesian",
            bagging_temperature=.5, task_type="GPU", devices="0",
            random_seed=seed, allow_writing_files=False, verbose=0,
        )
        model.fit(features.loc[train], label[train], cat_features=cat_indices)
        valid_members.append(model.predict(features.loc[valid]))
        train_members.append(model.predict(features.loc[train]))
        print(f"Previous-season state complete: year={args.valid_year} seed={seed}", flush=True)
    prediction = np.zeros(len(raw), dtype=float)
    prediction[train] = np.mean(train_members, axis=0)
    prediction[valid] = np.mean(valid_members, axis=0)
    direction = prediction[valid] - frozen_prediction_center(prediction, raw, train, valid)

    with np.load(root / "outputs/v23_oof_predictions.npz") as z:
        fold = z["season"] == args.valid_year
        y = z["target"][fold].astype(float)
        base = z["blended"][fold].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v23 OOF rows do not align")
    rows = raw.loc[valid].reset_index(drop=True)
    masks = segment_masks(rows)
    reports = []
    for gate_name, gate in (
        ("all", masks["all"]), ("regular", masks["regular"]),
        ("futures", masks["futures"]),
    ):
        gated = direction * gate
        for weight in np.arange(-.50, 1.501, .025):
            candidate = np.clip(base + weight * gated, .005, .995)
            gains = {
                name: bss(y[mask], candidate[mask]) - bss(y[mask], base[mask])
                for name, mask in masks.items() if mask.any()
            }
            reports.append({
                "gate": gate_name, "weight": float(weight), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_month": min(
                    gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                ),
            })
    reports.sort(
        key=lambda row: (min(row["min_half"], row["min_month"]), row["gains"]["all"]),
        reverse=True,
    )
    output = root / f"research/v23_previous_season_state_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        direction=direction.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "valid_year": args.valid_year, "feature_count": features.shape[1],
        "direction_mean": float(direction.mean()), "direction_std": float(direction.std()),
        "top": reports[:60],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
