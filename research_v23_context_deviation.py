"""Audit frozen pitcher-context target deviations over v23.

The signal is a prior-season, empirical-Bayes deviation from each pitcher's
overall rate.  Removing the pitcher level also removes most annual target-rate
drift, leaving a potentially transferable platoon/count interaction.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


CONTEXTS = {
    "hand": ("pitcher_id", "batter_hand"),
    "count": ("pitcher_id", "count_state"),
    "pressure_hand": ("pitcher_id", "pressure_state", "batter_hand"),
}
WINDOWS = (None, 1, 2, 3)
SHRINKS = (100., 300., 800.)
WEIGHTS = np.arange(-.20, .801, .025)


def masks(rows):
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
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def deviation(source, query, keys, shrink, value_column):
    pitcher = source.groupby("pitcher_id", observed=True)[value_column].agg(
        pitcher_sum="sum", pitcher_n="count",
    ).reset_index()
    pitcher["pitcher_rate"] = pitcher["pitcher_sum"] / pitcher["pitcher_n"]
    context = source.groupby(list(keys), observed=True)[value_column].agg(
        context_sum="sum", context_n="count",
    ).reset_index()
    context = context.merge(
        pitcher[["pitcher_id", "pitcher_rate"]],
        on="pitcher_id", how="left", validate="many_to_one",
    )
    context_rate = context["context_sum"] / context["context_n"]
    context["deviation"] = (
        context["context_n"] / (context["context_n"] + shrink)
        * (context_rate - context["pitcher_rate"])
    )
    left = query[list(keys)].copy()
    left["_order"] = np.arange(len(left))
    merged = left.merge(
        context[[*keys, "deviation"]], on=list(keys), how="left", sort=False,
    ).sort_values("_order")
    return merged["deviation"].fillna(0.).to_numpy(float)


def curve(target, base, signal, active):
    uncertainty = float(target[active].mean() * (1. - target[active].mean()))
    residual = target[active] - base[active]
    selected = signal[active]
    return (
        200000. * float(np.mean(residual * selected)) / uncertainty,
        100000. * float(np.mean(selected**2)) / uncertainty,
    )


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    data["count_state"] = (
        data["balls_before"].to_numpy(np.int16) * 3
        + data["strikes_before"].to_numpy(np.int16)
    )
    balls = data["balls_before"].to_numpy(np.int16)
    strikes = data["strikes_before"].to_numpy(np.int16)
    data["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    # A season-relative label is tested separately; it removes league-wide drift
    # before estimating within-pitcher context deviations.
    data["season_relative_target"] = (
        data["control_success"]
        - data.groupby(["season", "game_type"], observed=True)[
            "control_success"
        ].transform("mean")
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        oof = {key: source[key] for key in source.files}
    folds = {}
    for year in (2023, 2024):
        active = oof["season"] == year
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        target = oof["target"][active].astype(float)
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"v23 rows do not align for {year}")
        folds[year] = {
            "rows": rows, "target": target,
            "base": oof["blended"][active].astype(float), "masks": masks(rows),
        }

    signals = {}
    for year, item in folds.items():
        for window, source_game_type, value_column, context_name, shrink in itertools.product(
            WINDOWS, ("regular", "all"),
            ("control_success", "season_relative_target"),
            CONTEXTS, SHRINKS,
        ):
            source = data.loc[data["season"].lt(year)]
            if window is not None:
                source = source.loc[source["season"].ge(year - window)]
            if source_game_type == "regular":
                source = source.loc[source["game_type"].eq("R")]
            key = (window, source_game_type, value_column, context_name, shrink)
            raw_signal = deviation(
                source, item["rows"], CONTEXTS[context_name], shrink, value_column,
            )
            signals[(year, key)] = raw_signal
        print(f"Prepared context deviations for {year}", flush=True)

    approximate = []
    for key in itertools.product(
        WINDOWS, ("regular", "all"),
        ("control_success", "season_relative_target"), CONTEXTS, SHRINKS,
    ):
        for gate_name in ("regular", "all"):
            curves = {}
            for year, item in folds.items():
                gate = item["masks"][gate_name].astype(float)
                signal = gate * signals[(year, key)]
                curves[str(year)] = {
                    name: curve(item["target"], item["base"], signal, active)
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
                    "value": key[2], "context": key[3], "shrink": key[4],
                    "gate": gate_name, "weight": float(weight), "gains": gains,
                    "min_core": min(core), "min_segment": min(temporal),
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "mean_year": np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ]),
                })

    ranking_functions = {
        "maximin_core": lambda row: (
            row["min_core"], row["min_year"], row["mean_year"],
        ),
        "maximin_segment": lambda row: (
            row["min_segment"], row["min_year"], row["mean_year"],
        ),
        "maximin_year": lambda row: (
            row["min_year"], row["min_segment"], row["mean_year"],
        ),
        "best_mean": lambda row: (
            row["mean_year"], row["min_year"], row["min_segment"],
        ),
    }
    chosen = {}
    for ranking in ranking_functions.values():
        for row in sorted(approximate, key=ranking, reverse=True)[:300]:
            config = (
                row["window"], row["source_game_type"], row["value"],
                row["context"], row["shrink"], row["gate"], row["weight"],
            )
            chosen[config] = row

    reports = []
    for row in chosen.values():
        key = (
            row["window"], row["source_game_type"], row["value"],
            row["context"], row["shrink"],
        )
        gains = {}
        for year, item in folds.items():
            signal = (
                item["masks"][row["gate"]].astype(float) * signals[(year, key)]
            )
            candidate = np.clip(
                item["base"] + row["weight"] * signal, .005, .995,
            )
            gains[str(year)] = {
                name: bss(item["target"][active], candidate[active])
                - bss(item["target"][active], item["base"][active])
                for name, active in item["masks"].items() if active.any()
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
        reports.append({
            **{name: row[name] for name in (
                "window", "source_game_type", "value", "context", "shrink",
                "gate", "weight",
            )},
            "gains": gains, "min_core": min(core),
            "min_segment": min(temporal),
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "mean_year": np.mean([
                gains["2023"]["all"], gains["2024"]["all"],
            ]),
        })

    rankings = {
        name: sorted(reports, key=ranking, reverse=True)[:150]
        for name, ranking in ranking_functions.items()
    }
    output = root / "research/v23_context_deviation.json"
    output.write_text(json.dumps({"rankings": rankings}, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:15] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
