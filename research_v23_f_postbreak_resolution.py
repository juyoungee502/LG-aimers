"""Post-break F-only resolution model with three chronological transfers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


SEEDS = (4901, 4902, 4903)
CONTEXTS = {
    "count": ("count_state",),
    "count_runners": ("count_state", "runner_gate"),
    "count_hands": ("count_state", "pitcher_hand", "batter_hand"),
}
DROP_COLUMNS = (
    "season", "game_month", "pitcher_id", "batter_id", "pitcher_team_id",
    "batter_team_id", "team_matchup",
)


def context_frame(raw):
    return pd.DataFrame({
        "count_state": raw["balls_before"] * 3 + raw["strikes_before"],
        "runner_gate": raw["num_runners_on"].gt(0).astype(np.int8),
        "pitcher_hand": raw["pitcher_hand"],
        "batter_hand": raw["batter_hand"],
    }, index=raw.index)


def centered_label(target, context, train, keys):
    work = context.loc[train, list(keys)].copy()
    work["target"] = target[train]
    center = work.groupby(list(keys), observed=True)["target"].transform("mean")
    return target[train] - center.to_numpy(float)


def frozen_prediction_center(train_prediction, context, train, valid, keys):
    source = context.loc[train, list(keys)].copy()
    source["prediction"] = train_prediction
    table = source.groupby(list(keys), observed=True)["prediction"].mean().reset_index()
    query = context.loc[valid, list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    return query["prediction"].fillna(float(source["prediction"].mean())).to_numpy(float)


def block_masks(rows):
    position = np.arange(len(rows))
    return {
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
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features).drop(columns=list(DROP_COLUMNS))
    context = context_frame(raw)
    seasons = raw["season"].to_numpy(np.int16)
    futures = raw["game_type"].eq("F").to_numpy()
    positions = np.arange(len(raw))
    f23 = np.flatnonzero((seasons == 2023) & futures)
    f24 = np.flatnonzero((seasons == 2024) & futures)
    transfers = {
        "23h1_to_h2": (
            np.isin(positions, f23[:len(f23) // 2]),
            np.isin(positions, f23[len(f23) // 2:]),
        ),
        "23_to_24": ((seasons == 2023) & futures, (seasons == 2024) & futures),
        "24h1_to_h2": (
            np.isin(positions, f24[:len(f24) // 2]),
            np.isin(positions, f24[len(f24) // 2:]),
        ),
    }
    with np.load(root / "outputs/v23_oof_predictions.npz") as z:
        v23 = {key: z[key] for key in z.files}
    year_bases = {}
    for year in (2023, 2024):
        fold = v23["season"] == year
        year_rows = raw.loc[seasons == year].reset_index(drop=True)
        year_f = year_rows["game_type"].eq("F").to_numpy()
        year_bases[year] = v23["blended"][fold][year_f].astype(float)
        if not np.allclose(v23["target"][fold][year_f], target[(seasons == year) & futures]):
            raise ValueError(f"v23 F rows do not align for {year}")

    directions = {}
    reports = []
    for mode_offset, (mode, keys) in enumerate(CONTEXTS.items()):
        directions[mode] = {}
        for transfer_offset, (transfer, (train, valid)) in enumerate(transfers.items()):
            label = centered_label(target, context, train, keys)
            valid_members = []
            train_members = []
            for seed in SEEDS:
                model = CatBoostRegressor(
                    iterations=900, depth=5, learning_rate=.025,
                    loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=120.,
                    random_strength=1., border_count=32, bootstrap_type="Bayesian",
                    bagging_temperature=.5, task_type="GPU", devices="0",
                    random_seed=seed + 100 * mode_offset + 10 * transfer_offset,
                    allow_writing_files=False, verbose=0,
                )
                model.fit(features.loc[train], label)
                valid_members.append(model.predict(features.loc[valid]))
                train_members.append(model.predict(features.loc[train]))
            valid_prediction = np.mean(valid_members, axis=0)
            train_prediction = np.mean(train_members, axis=0)
            direction = valid_prediction - frozen_prediction_center(
                train_prediction, context, train, valid, keys,
            )
            directions[mode][transfer] = direction.astype(np.float32)
            valid_rows = raw.loc[valid].reset_index(drop=True)
            year = int(valid_rows["season"].iloc[0])
            if transfer == "23h1_to_h2":
                base = year_bases[2023][len(f23) // 2:]
            elif transfer == "24h1_to_h2":
                base = year_bases[2024][len(f24) // 2:]
            else:
                base = year_bases[2024]
            y = target[valid]
            masks = block_masks(valid_rows)
            for weight in np.arange(-.25, 1.501, .025):
                candidate = np.clip(base + weight * direction, .005, .995)
                gains = {
                    name: bss(y[mask], candidate[mask]) - bss(y[mask], base[mask])
                    for name, mask in masks.items() if mask.any()
                }
                reports.append({
                    "mode": mode, "transfer": transfer, "weight": float(weight),
                    "gains": gains,
                    "min_segment": min(value for name, value in gains.items() if name != "all"),
                })
            print(
                f"F post-break complete: mode={mode} transfer={transfer} "
                f"train={train.sum()} valid={valid.sum()}", flush=True,
            )

    joint = []
    for mode in CONTEXTS:
        for weight in np.arange(0., 1.501, .025):
            selected = [
                row for row in reports
                if row["mode"] == mode and abs(row["weight"] - weight) < 1e-8
            ]
            by_transfer = {row["transfer"]: row["gains"] for row in selected}
            joint.append({
                "mode": mode, "weight": float(weight), "gains": by_transfer,
                "min_segment": min(row["min_segment"] for row in selected),
                "min_transfer": min(row["gains"]["all"] for row in selected),
                "mean_transfer": np.mean([row["gains"]["all"] for row in selected]),
            })
    joint.sort(
        key=lambda row: (row["min_segment"], row["min_transfer"], row["mean_transfer"]),
        reverse=True,
    )
    output = root / "research/v23_f_postbreak_resolution.npz"
    np.savez_compressed(
        output,
        names=np.asarray(list(CONTEXTS)),
        direction_23_to_24=np.column_stack([
            directions[mode]["23_to_24"] for mode in CONTEXTS
        ]),
        reports_json=np.asarray(json.dumps(joint)),
    )
    print(json.dumps({"top": joint[:80]}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
