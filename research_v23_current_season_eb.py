"""Audit row-local current-season empirical-Bayes anchors over v23."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import TARGET_COL, engineer_features, training_history_arrays
from research_inferred_pitch_priors import bss


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
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(
        raw, *bases, global_prior=float(target.mean()),
    )
    season_n = features["pitcher_season_n"].to_numpy(float)
    season_success = features["pitcher_season_success_count"].to_numpy(float)
    batter_n = features["batter_season_n"].to_numpy(float)
    batter_success = features["batter_season_success_count"].to_numpy(float)
    career = features["pitcher_prior_success_rate"].to_numpy(float)
    batter_career = features["batter_prior_success_rate"].to_numpy(float)
    recent = {
        "prev1": raw["asof_pitcher_prev1_game_success_rate"].to_numpy(float),
        "prev3": raw["asof_pitcher_prev3_game_success_rate"].to_numpy(float),
        "prev5": raw["asof_pitcher_prev5_game_success_rate"].to_numpy(float),
    }
    recent["blend135"] = (
        .15 * recent["prev1"] + .35 * recent["prev3"] + .50 * recent["prev5"]
    )
    recent["blend35"] = .35 * recent["prev3"] + .65 * recent["prev5"]
    anchors = {}
    for strength in (10., 25., 50., 100., 200.):
        anchors[f"career_s{int(strength)}"] = (
            season_success + strength * career
        ) / (season_n + strength)
        anchors[f"batter_career_s{int(strength)}"] = (
            batter_success + strength * batter_career
        ) / (batter_n + strength)
        for name, value in recent.items():
            finite = np.where(np.isfinite(value), value, career)
            anchors[f"{name}_s{int(strength)}"] = (
                season_success + strength * finite
            ) / (season_n + strength)
            anchors[f"career_{name}_s{int(strength)}"] = (
                season_success + strength * (.5 * finite + .5 * career)
            ) / (season_n + strength)
    for name, value in recent.items():
        anchors[f"recent_{name}"] = np.where(np.isfinite(value), value, career)

    seasons = raw["season"].to_numpy(np.int16)
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        oof = {key: source[key] for key in source.files}
    exposure_gates = {
        "all": np.ones(len(raw), dtype=float),
        "regular": raw["game_type"].eq("R").to_numpy(float),
        "regular_n25": raw["game_type"].eq("R").to_numpy(float)
        * (season_n / (season_n + 25.)),
        "regular_n50": raw["game_type"].eq("R").to_numpy(float)
        * (season_n / (season_n + 50.)),
        "regular_n100": raw["game_type"].eq("R").to_numpy(float)
        * (season_n / (season_n + 100.)),
        "regular_decay100": raw["game_type"].eq("R").to_numpy(float)
        * (100. / (season_n + 100.)),
        "futures": raw["game_type"].eq("F").to_numpy(float),
    }

    reports = []
    for anchor_name, anchor in anchors.items():
        for gate_name, full_gate in exposure_gates.items():
            curves = {}
            for year in (2023, 2024):
                fold = oof["season"] == year
                rows = raw.loc[seasons == year].reset_index(drop=True)
                y = oof["target"][fold].astype(float)
                base = oof["blended"][fold].astype(float)
                if not np.allclose(y, target[seasons == year]):
                    raise ValueError(f"v23 rows do not align for {year}")
                direction = full_gate[seasons == year] * (
                    np.clip(anchor[seasons == year], .01, .99) - base
                )
                curves[str(year)] = {}
                for name, mask in masks(rows).items():
                    if not mask.any():
                        continue
                    uncertainty = float(y[mask].mean() * (1. - y[mask].mean()))
                    curves[str(year)][name] = (
                        200000. * float(np.mean(
                            (y[mask] - base[mask]) * direction[mask]
                        )) / uncertainty,
                        100000. * float(np.mean(direction[mask] ** 2)) / uncertainty,
                    )
            for weight in np.arange(-.30, .301, .025):
                gains = {
                    year: {
                        name: linear * weight - quadratic * weight**2
                        for name, (linear, quadratic) in year_curves.items()
                    }
                    for year, year_curves in curves.items()
                }
                temporal = [
                    value for year_gains in gains.values()
                    for name, value in year_gains.items() if name != "all"
                ]
                reports.append({
                    "anchor": anchor_name, "gate": gate_name,
                    "geometry": "probability", "weight": float(weight),
                    "gains": gains, "min_segment": min(temporal),
                    "min_year": min(
                        gains["2023"]["all"], gains["2024"]["all"],
                    ),
                    "mean_year": np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ]),
                })
    nonzero = [row for row in reports if abs(row["weight"]) > 1e-8]
    rankings = {
        "maximin_segment": sorted(
            nonzero,
            key=lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
            reverse=True,
        )[:100],
        "maximin_year": sorted(
            nonzero,
            key=lambda row: (row["min_year"], row["min_segment"], row["mean_year"]),
            reverse=True,
        )[:100],
        "best_mean": sorted(
            nonzero,
            key=lambda row: (row["mean_year"], row["min_year"], row["min_segment"]),
            reverse=True,
        )[:100],
    }
    output = root / "research/v23_current_season_eb.json"
    output.write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:15] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
