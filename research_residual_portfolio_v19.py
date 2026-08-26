"""Screen frozen empirical-Bayes residual tables across four time transfers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SPECS = {
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "pitcher_count": ["pitcher_id", "count_state"],
    "pitcher_bhand": ["pitcher_id", "batter_hand"],
    "pitcher_game": ["pitcher_id", "game_type"],
    "pitcher_inning": ["pitcher_id", "inning_bucket"],
    "pitcher_baseout": ["pitcher_id", "base_out_state"],
    "pitcher_month": ["pitcher_id", "game_month"],
    "batter_count": ["batter_id", "count_state"],
    "batter_phand": ["batter_id", "pitcher_hand"],
    "batter_game": ["batter_id", "game_type"],
    "pitcher_team_count": ["pitcher_team_id", "count_state"],
    "batter_team_count": ["batter_team_id", "count_state"],
    "count_hand": ["count_state", "pitcher_hand", "batter_hand"],
    "count_game": ["count_state", "game_type"],
    "baseout_count": ["base_out_state", "count_state"],
}


def prepare(frame):
    result = frame.copy()
    result["count_state"] = (
        result["balls_before"] * 3 + result["strikes_before"]
    ).astype(np.int8)
    result["base_out_state"] = (
        result["base_state"].map({
            "___": 0, "1__": 1, "_2_": 2, "__3": 3,
            "12_": 4, "1_3": 5, "_23": 6, "123": 7,
        }).fillna(-1).astype(np.int8) * 3 + result["outs_before"]
    )
    result["inning_bucket"] = np.select(
        [result["inning"] <= 3, result["inning"] <= 6], [0, 1], default=2,
    ).astype(np.int8)
    return result


def build_table(frame, residual, keys, shrink):
    work = frame[keys].copy()
    work["residual"] = np.asarray(residual, dtype=np.float64)
    table = work.groupby(keys, observed=True, sort=False)["residual"].agg(
        ["sum", "size"]
    ).reset_index()
    table["value"] = table["sum"] / (table["size"] + shrink)
    source_value = apply_table(frame, table, keys)
    table["value"] -= float(source_value.mean())
    return table[keys + ["value"]]


def apply_table(frame, table, keys):
    left = frame[keys].copy()
    left["_order"] = np.arange(len(left))
    merged = left.merge(table, on=keys, how="left", sort=False).sort_values("_order")
    return merged["value"].fillna(0.).to_numpy(np.float64)


def coefficients(target, base, value, active):
    value = np.where(active, value, 0.)
    uncertainty = float(target.mean() * (1. - target.mean()))
    residual = target - base
    linear = 100000. * np.mean(2. * residual * value) / uncertainty
    quadratic = 100000. * np.mean(value * value) / uncertainty
    return float(linear), float(quadratic)


def gain(pair, weight):
    linear, quadratic = pair
    return weight * linear - weight * weight * quadratic


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    rows = {
        year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
        for year in (2023, 2024)
    }
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        y = oof["target"][mask].astype(np.float64)
        base = oof["blended"][mask].astype(np.float64)
        if not np.allclose(y, rows[year]["control_success"]):
            raise ValueError(f"v19 OOF rows differ for {year}")
        folds[year] = {"y": y, "base": base, "residual": y - base}

    midpoint = {year: len(rows[year]) // 2 for year in rows}
    reports = []
    for name, keys in SPECS.items():
        for shrink in (100., 200., 400., 800., 1600., 3200., 6400., 12800.):
            # Three frozen sources mimic increasingly close deployment:
            # early-2023 -> late-2023, 2023 -> both 2024 halves, and
            # early-2024 -> late-2024.
            m23 = midpoint[2023]
            m24 = midpoint[2024]
            table_23h1 = build_table(
                rows[2023].iloc[:m23], folds[2023]["residual"][:m23], keys, shrink,
            )
            table_23 = build_table(
                rows[2023], folds[2023]["residual"], keys, shrink,
            )
            table_24h1 = build_table(
                rows[2024].iloc[:m24], folds[2024]["residual"][:m24], keys, shrink,
            )
            target_blocks = [
                (rows[2023].iloc[m23:], folds[2023]["y"][m23:],
                 folds[2023]["base"][m23:], table_23h1, "2023_h2"),
                (rows[2024].iloc[:m24], folds[2024]["y"][:m24],
                 folds[2024]["base"][:m24], table_23, "2024_h1"),
                (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                 folds[2024]["base"][m24:], table_23, "2024_h2"),
                (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                 folds[2024]["base"][m24:], table_24h1, "2024_h1_to_h2"),
            ]
            pairs = {}
            for frame, y, base, table, label in target_blocks:
                value = apply_table(frame, table, keys)
                active = frame["game_type"].eq("R").to_numpy()
                pairs[label] = coefficients(y, base, value, active)
            for weight in np.arange(-1., 2.001, .025):
                values = {label: gain(pair, weight) for label, pair in pairs.items()}
                reports.append({
                    "name": name, "keys": keys, "shrink": shrink,
                    "weight": float(weight), "gains": values,
                    "min_transfer": min(values.values()),
                    "mean_transfer": float(np.mean(list(values.values()))),
                })
    reports.sort(
        key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True,
    )
    output = root / "research/residual_portfolio_v19.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:80], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
