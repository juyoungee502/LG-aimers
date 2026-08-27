"""Audit heavily shrunk pitcher/batter residual effects over v23."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = {
    "pitcher": ("pitcher_id",),
    "batter": ("batter_id",),
    "batter_pitcher_hand": ("batter_id", "pitcher_hand"),
    "pitcher_batter_hand": ("pitcher_id", "batter_hand"),
}
SHRINKS = (1000., 2500., 5000., 10000., 20000., 50000., 100000.)


def bss_gain(target, base, correction):
    reference = float(target.mean() * (1.0 - target.mean()))
    residual = target - base
    return float(100000.0 * np.mean(
        2.0 * residual * correction - correction * correction
    ) / reference)


def fit_table(frame, residual, keys, shrink):
    source = frame.loc[frame["game_type"].eq("R"), list(keys)].copy()
    source["residual"] = residual[frame["game_type"].eq("R").to_numpy()]
    table = source.groupby(list(keys), observed=True)["residual"].agg(
        residual_sum="sum", residual_n="count",
    ).reset_index()
    table["effect"] = table["residual_sum"] / (table["residual_n"] + shrink)
    return table[[*keys, "effect"]]


def apply_table(frame, table, keys):
    query = frame[list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    effect = query["effect"].fillna(0.0).to_numpy(float, copy=True)
    effect[~frame["game_type"].eq("R").to_numpy()] = 0.0
    return effect


def fit_stats(frame, residual, keys):
    regular = frame["game_type"].eq("R").to_numpy()
    source = frame.loc[regular, list(keys)].copy()
    source["residual"] = residual[regular]
    return source.groupby(list(keys), observed=True)["residual"].agg(
        residual_sum="sum", residual_n="count",
    ).reset_index()


def apply_stats(frame, table, keys):
    query = frame[list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    total = query["residual_sum"].fillna(0.0).to_numpy(float, copy=True)
    count = query["residual_n"].fillna(0.0).to_numpy(float, copy=True)
    regular = frame["game_type"].eq("R").to_numpy()
    total[~regular] = 0.0
    count[~regular] = 0.0
    return total, count


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "batter_id",
            "pitcher_hand", "batter_hand",
        ], encoding="utf-8-sig", low_memory=False,
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        folds[year] = {
            "frame": frame,
            "target": oof["target"][mask].astype(float),
            "base": oof["blended"][mask].astype(float),
        }
        folds[year]["residual"] = folds[year]["target"] - folds[year]["base"]
    cuts = {year: np.linspace(0, len(item["frame"]), 5, dtype=int)
            for year, item in folds.items()}
    source_specs = {
        "23h1": (2023, slice(0, cuts[2023][2])),
        "23full": (2023, slice(None)),
        "24h1": (2024, slice(0, cuts[2024][2])),
    }
    block_specs = {
        "23h1_to_h2": (2023, slice(cuts[2023][2], None), "23h1"),
        "24h1_to_h2": (2024, slice(cuts[2024][2], None), "24h1"),
    }
    for quarter in range(4):
        block_specs[f"23_to_24q{quarter + 1}"] = (
            2024, slice(cuts[2024][quarter], cuts[2024][quarter + 1]), "23full",
        )

    candidates = []
    for group_name, keys in GROUPS.items():
        tables = {}
        for source_name, (year, section) in source_specs.items():
            source = folds[year]
            tables[source_name] = fit_stats(
                source["frame"].iloc[section].reset_index(drop=True),
                source["residual"][section], keys,
            )
        matched = {}
        for label, (year, section, source_name) in block_specs.items():
            matched[label] = apply_stats(
                folds[year]["frame"].iloc[section].reset_index(drop=True),
                tables[source_name], keys,
            )
        for shrink in SHRINKS:
            directions = {
                label: total / (count + shrink)
                for label, (total, count) in matched.items()
            }
            for weight in np.arange(-1.0, 5.001, .10):
                gains = {}
                for label, (year, section, _source_name) in block_specs.items():
                    item = folds[year]
                    gains[label] = bss_gain(
                        item["target"][section], item["base"][section],
                        weight * directions[label],
                    )
                candidates.append({
                    "group": group_name, "keys": list(keys),
                    "shrink": shrink, "weight": float(weight), "gains": gains,
                    "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
    candidates.sort(
        key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True,
    )
    output = root / "research/v23_entity_residual.json"
    output.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(json.dumps({"top": candidates[:100]}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
