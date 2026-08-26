"""Screen robust combinations of no-month, full-history, and recent command axes.

The command features are frozen from prior seasons.  At inference time the gate uses
only the pitcher's current-season exposure already present in the row-level as-of
features, so it does not aggregate or otherwise inspect test rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }


def add_pitcher_season_exposure(rows):
    values = np.zeros(len(rows), dtype=np.float32)
    end_n = {}
    seasons = rows["season"].to_numpy()
    for season in np.sort(rows["season"].unique()):
        positions = np.flatnonzero(seasons == season)
        block = rows.iloc[positions]
        base = block["pitcher_id"].map(end_n).fillna(0.0).to_numpy(float)
        values[positions] = np.maximum(
            0.0, block["asof_pitcher_n"].fillna(0.0).to_numpy(float) - base,
        )
        last = block.groupby("pitcher_id", observed=True, sort=False).tail(1)
        end_n.update(zip(
            last["pitcher_id"].astype(int).tolist(),
            (last["asof_pitcher_n"].fillna(0.0) + 1.0).tolist(),
        ))
    result = rows.copy()
    result["pitcher_season_n"] = values
    return result


def quadratic_curve(target, base, axes, mask):
    reference = float(target[mask].mean() * (1.0 - target[mask].mean()))
    residual = target[mask] - base[mask]
    selected = [axis[mask] for axis in axes]
    linear = [
        200000.0 * float(np.mean(axis * residual)) / reference
        for axis in selected
    ]
    quadratic = [
        [100000.0 * float(np.mean(left * right)) / reference for right in selected]
        for left in selected
    ]
    return {"linear": linear, "quadratic": quadratic}


def approximate_gain(curve, weights):
    linear = np.asarray(curve["linear"])
    quadratic = np.asarray(curve["quadratic"])
    weights = np.asarray(weights)
    return float(linear @ weights - weights @ quadratic @ weights)


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data" / "train.csv",
        usecols=["season", "game_month", "pitcher_id", "asof_pitcher_n"],
        encoding="utf-8-sig", low_memory=False,
    )
    data = add_pitcher_season_exposure(data)
    years = {}
    for year in (2023, 2024):
        with np.load(root / "research" / f"v23_trackman_no_month_{year}.npz") as z:
            target = z["target"].astype(float)
            base = z["base"].astype(float)
            no_month = z["direction"].astype(float)
        with np.load(root / "research" / f"v23_prior_command_context_{year}.npz") as z:
            full_command = z["command_direction"].astype(float)
        with np.load(root / "research" / f"v23_prior_command_context_{year}_w1.npz") as z:
            recent_command = z["command_direction"].astype(float)
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        years[year] = {
            "target": target,
            "base": base,
            "no_month": no_month,
            "full": full_command,
            "recent": recent_command,
            "rows": rows,
            "masks": masks(rows),
        }

    gate_functions = {
        "all": lambda rows: np.ones(len(rows)),
        "through_july": lambda rows: (rows["game_month"].to_numpy() <= 7).astype(float),
        "pitch_n_400": lambda rows: (rows["pitcher_season_n"].to_numpy() <= 400).astype(float),
        "pitch_n_600": lambda rows: (rows["pitcher_season_n"].to_numpy() <= 600).astype(float),
        "pitch_n_800": lambda rows: (rows["pitcher_season_n"].to_numpy() <= 800).astype(float),
        "pitch_decay_300": lambda rows: 300.0 / (300.0 + rows["pitcher_season_n"].to_numpy()),
        "pitch_decay_600": lambda rows: 600.0 / (600.0 + rows["pitcher_season_n"].to_numpy()),
    }
    # Pair choices are deliberately limited to interpretable deployment policies.
    gate_pairs = [
        ("pitch_n_400", "all"),
        ("pitch_n_600", "all"),
        ("pitch_n_800", "all"),
        ("through_july", "all"),
        ("pitch_n_600", "pitch_n_600"),
        ("through_july", "through_july"),
        ("pitch_decay_300", "all"),
        ("pitch_decay_600", "all"),
        ("pitch_decay_600", "pitch_decay_600"),
        ("all", "all"),
    ]
    reports = []
    for full_gate_name, recent_gate_name in gate_pairs:
        curves = {}
        for year, item in years.items():
            full_gate = gate_functions[full_gate_name](item["rows"])
            recent_gate = gate_functions[recent_gate_name](item["rows"])
            derivative = item["base"] * (1.0 - item["base"])
            axes = [
                derivative * item["no_month"],
                derivative * full_gate * item["full"],
                derivative * recent_gate * item["recent"],
            ]
            curves[str(year)] = {
                name: quadratic_curve(item["target"], item["base"], axes, mask)
                for name, mask in item["masks"].items()
            }
        for no_month_weight in np.arange(.50, 1.351, .05):
            for full_weight in np.arange(-.20, 1.201, .10):
                for recent_weight in np.arange(-.20, 1.001, .10):
                    weights = [no_month_weight, full_weight, recent_weight]
                    gains = {}
                    segments = []
                    for year in ("2023", "2024"):
                        gains[year] = {}
                        for name, values in curves[year].items():
                            gain = approximate_gain(values, weights)
                            gains[year][name] = gain
                            if name != "all":
                                segments.append(gain)
                    reports.append({
                        "full_gate": full_gate_name,
                        "recent_gate": recent_gate_name,
                        "weights": weights,
                        "gains": gains,
                        "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                        "min_segment": min(segments),
                        "mean_year": (gains["2023"]["all"] + gains["2024"]["all"]) / 2,
                    })

    # Preserve candidates from three useful rankings before exact sigmoid scoring.
    chosen = {}
    ranking_keys = (
        lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
        lambda row: (row["min_year"], row["min_segment"], row["mean_year"]),
        lambda row: (row["mean_year"], row["min_segment"], row["min_year"]),
    )
    for ranking in ranking_keys:
        for report in sorted(reports, key=ranking, reverse=True)[:120]:
            key = (
                report["full_gate"], report["recent_gate"],
                *(round(value, 4) for value in report["weights"]),
            )
            chosen[key] = report

    exact_reports = []
    for report in chosen.values():
        gains = {}
        segments = []
        for year, item in years.items():
            full_gate = gate_functions[report["full_gate"]](item["rows"])
            recent_gate = gate_functions[report["recent_gate"]](item["rows"])
            weights = report["weights"]
            candidate = sigmoid(
                logit(item["base"])
                + weights[0] * item["no_month"]
                + weights[1] * full_gate * item["full"]
                + weights[2] * recent_gate * item["recent"]
            )
            gains[str(year)] = {}
            for name, mask in item["masks"].items():
                gain = bss(item["target"][mask], candidate[mask]) - bss(
                    item["target"][mask], item["base"][mask],
                )
                gains[str(year)][name] = gain
                if name != "all":
                    segments.append(gain)
        exact_reports.append({
            "full_gate": report["full_gate"],
            "recent_gate": report["recent_gate"],
            "weights": report["weights"],
            "gains": gains,
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "min_segment": min(segments),
            "mean_year": (gains["2023"]["all"] + gains["2024"]["all"]) / 2,
        })

    output_rankings = {}
    names = ("maximin_segment", "maximin_year", "best_mean")
    for name, ranking in zip(names, ranking_keys):
        output_rankings[name] = sorted(exact_reports, key=ranking, reverse=True)[:40]
    output = root / "research" / "v23_joint_command_history.json"
    output.write_text(json.dumps(output_rankings, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:10] for name, rows in output_rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
