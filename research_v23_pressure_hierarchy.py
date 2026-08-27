"""Audit time-safe hierarchical pitcher x pressure x batter-hand deviations.

The child rate is shrunk toward either the pitcher's overall rate or the
pitcher x batter-hand parent rate.  Every lookup table is built only from
seasons before the validation season, so the experiment mirrors submission
inference and never aggregates validation/test rows.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v23_context_deviation import curve, masks


WINDOWS = (None, 1, 2, 3, 4)
SHRINKS = (75., 150., 220., 400., 800., 1200.)
WEIGHTS = np.arange(-.10, .701, .025)


def hierarchical_deviation(
    source: pd.DataFrame,
    query: pd.DataFrame,
    parent_keys: tuple[str, ...],
    child_keys: tuple[str, ...],
    shrink: float,
    value_column: str,
) -> np.ndarray:
    parent = source.groupby(list(parent_keys), observed=True)[value_column].agg(
        parent_sum="sum", parent_n="count",
    ).reset_index()
    parent["parent_rate"] = parent["parent_sum"] / parent["parent_n"]
    child = source.groupby(list(child_keys), observed=True)[value_column].agg(
        child_sum="sum", child_n="count",
    ).reset_index()
    child = child.merge(
        parent[[*parent_keys, "parent_rate"]],
        on=list(parent_keys), how="left", validate="many_to_one",
    )
    child_rate = child["child_sum"] / child["child_n"]
    child["deviation"] = (
        child["child_n"] / (child["child_n"] + shrink)
        * (child_rate - child["parent_rate"])
    )
    left = query[list(child_keys)].copy()
    left["_order"] = np.arange(len(left))
    merged = left.merge(
        child[[*child_keys, "deviation"]],
        on=list(child_keys), how="left", sort=False,
    ).sort_values("_order")
    return merged["deviation"].fillna(0.).to_numpy(float)


def report(target, base, signal, weight, segment_masks):
    candidate = np.clip(base + weight * signal, .005, .995)
    return {
        name: bss(target[active], candidate[active])
        - bss(target[active], base[active])
        for name, active in segment_masks.items() if active.any()
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    balls = data["balls_before"].to_numpy(np.int16)
    strikes = data["strikes_before"].to_numpy(np.int16)
    data["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    data["season_relative_target"] = (
        data["control_success"]
        - data.groupby(["season", "game_type"], observed=True)[
            "control_success"
        ].transform("mean")
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}

    folds = {}
    for year in (2023, 2024):
        active = oof["season"] == year
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        target = oof["target"][active].astype(float)
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"v23 rows do not align for {year}")
        folds[year] = {
            "rows": rows,
            "target": target,
            "base": oof["blended"][active].astype(float),
            "masks": masks(rows),
        }

    parent_options = {
        "pitcher": ("pitcher_id",),
        "pitcher_hand": ("pitcher_id", "batter_hand"),
    }
    child_keys = ("pitcher_id", "batter_hand", "pressure_state")
    signals = {}
    configs = list(itertools.product(
        WINDOWS, ("regular", "all"),
        ("control_success", "season_relative_target"),
        parent_options, SHRINKS,
    ))
    for year, item in folds.items():
        for window, source_game_type, value_column, parent_name, shrink in configs:
            source = data.loc[data["season"].lt(year)]
            if window is not None:
                source = source.loc[source["season"].ge(year - window)]
            if source_game_type == "regular":
                source = source.loc[source["game_type"].eq("R")]
            key = (window, source_game_type, value_column, parent_name, shrink)
            signals[(year, key)] = hierarchical_deviation(
                source, item["rows"], parent_options[parent_name], child_keys,
                shrink, value_column,
            )
        print(f"Prepared hierarchical pressure deviations for {year}", flush=True)

    approximate = []
    for key in configs:
        for gate_name in ("all", "regular"):
            curves = {}
            for year, item in folds.items():
                signal = (
                    item["masks"][gate_name].astype(float)
                    * signals[(year, key)]
                )
                curves[str(year)] = {
                    name: curve(
                        item["target"], item["base"], signal, active,
                    )
                    for name, active in item["masks"].items() if active.any()
                }
            for weight in WEIGHTS:
                gains = {
                    year: {
                        name: linear * weight - quadratic * weight**2
                        for name, (linear, quadratic) in year_curves.items()
                    }
                    for year, year_curves in curves.items()
                }
                core = [
                    value for year_gains in gains.values()
                    for name, value in year_gains.items()
                    if name in (
                        "all", "first_half", "second_half", "months_3_5",
                        "months_6_7", "months_8_11", "regular", "futures",
                    )
                ]
                temporal = [
                    value for year_gains in gains.values()
                    for name, value in year_gains.items()
                    if name not in ("all", "regular", "futures")
                ]
                approximate.append({
                    "window": key[0], "source_game_type": key[1],
                    "value": key[2], "parent": key[3], "shrink": key[4],
                    "gate": gate_name, "weight": float(weight), "gains": gains,
                    "min_core": min(core), "min_temporal": min(temporal),
                    "min_year": min(
                        gains["2023"]["all"], gains["2024"]["all"],
                    ),
                    "mean_year": float(np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ])),
                })

    ranking_functions = {
        "maximin_core": lambda row: (
            row["min_core"], row["min_year"], row["mean_year"],
        ),
        "maximin_temporal": lambda row: (
            row["min_temporal"], row["min_year"], row["mean_year"],
        ),
        "maximin_year": lambda row: (
            row["min_year"], row["min_temporal"], row["mean_year"],
        ),
        "best_mean": lambda row: (
            row["mean_year"], row["min_year"], row["min_temporal"],
        ),
    }
    chosen = {}
    for ranking in ranking_functions.values():
        for row in sorted(approximate, key=ranking, reverse=True)[:300]:
            config = (
                row["window"], row["source_game_type"], row["value"],
                row["parent"], row["shrink"], row["gate"], row["weight"],
            )
            chosen[config] = row

    reports = []
    for row in chosen.values():
        key = (
            row["window"], row["source_game_type"], row["value"],
            row["parent"], row["shrink"],
        )
        gains = {}
        for year, item in folds.items():
            signal = (
                item["masks"][row["gate"]].astype(float)
                * signals[(year, key)]
            )
            gains[str(year)] = report(
                item["target"], item["base"], signal, row["weight"],
                item["masks"],
            )
        core = [
            value for year_gains in gains.values()
            for name, value in year_gains.items()
            if name in (
                "all", "first_half", "second_half", "months_3_5",
                "months_6_7", "months_8_11", "regular", "futures",
            )
        ]
        temporal = [
            value for year_gains in gains.values()
            for name, value in year_gains.items()
            if name not in ("all", "regular", "futures")
        ]
        reports.append({
            **{name: row[name] for name in (
                "window", "source_game_type", "value", "parent", "shrink",
                "gate", "weight",
            )},
            "gains": gains, "min_core": min(core),
            "min_temporal": min(temporal),
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "mean_year": float(np.mean([
                gains["2023"]["all"], gains["2024"]["all"],
            ])),
        })
    rankings = {
        name: sorted(reports, key=ranking, reverse=True)[:100]
        for name, ranking in ranking_functions.items()
    }
    output = root / "research/v23_pressure_hierarchy.json"
    output.write_text(json.dumps({"rankings": rankings}, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:12] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
