"""Jointly screen no-month replacement and early-season command context."""
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
    rows = rows.copy()
    rows["pitcher_season_n"] = values
    return rows


def curve(target, base, first, second, mask):
    reference = float(target[mask].mean() * (1.0 - target[mask].mean()))
    residual = target[mask] - base[mask]
    first = first[mask]
    second = second[mask]
    return {
        "l1": 200000.0 * float(np.mean(first * residual)) / reference,
        "l2": 200000.0 * float(np.mean(second * residual)) / reference,
        "q11": 100000.0 * float(np.mean(first**2)) / reference,
        "q22": 100000.0 * float(np.mean(second**2)) / reference,
        "q12": 100000.0 * float(np.mean(first * second)) / reference,
    }


def curve_gain(values, first_weight, second_weight):
    return (
        values["l1"] * first_weight + values["l2"] * second_weight
        - values["q11"] * first_weight**2
        - values["q22"] * second_weight**2
        - 2.0 * values["q12"] * first_weight * second_weight
    )


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
            no_month_direction = z["direction"].astype(float)
        with np.load(root / "research" / f"v23_prior_command_context_{year}.npz") as z:
            command_direction = z["command_direction"].astype(float)
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        years[year] = {
            "target": target, "base": base, "no_month": no_month_direction,
            "command": command_direction, "rows": rows, "masks": masks(rows),
        }

    gates = {
        "through_july": lambda rows: (rows["game_month"].to_numpy() <= 7).astype(float),
        "spring_only": lambda rows: (rows["game_month"].to_numpy() <= 5).astype(float),
        "july_half": lambda rows: np.where(
            rows["game_month"].to_numpy() <= 5, 1.0,
            np.where(rows["game_month"].to_numpy() <= 7, .5, 0.0),
        ),
        "late_quarter": lambda rows: np.where(
            rows["game_month"].to_numpy() <= 5, 1.0,
            np.where(rows["game_month"].to_numpy() <= 7, .75, .25),
        ),
        "pitch_n_300": lambda rows: (rows["pitcher_season_n"].to_numpy() <= 300).astype(float),
        "pitch_n_600": lambda rows: (rows["pitcher_season_n"].to_numpy() <= 600).astype(float),
        "pitch_decay_200": lambda rows: 200.0 / (200.0 + rows["pitcher_season_n"].to_numpy()),
        "pitch_decay_500": lambda rows: 500.0 / (500.0 + rows["pitcher_season_n"].to_numpy()),
        "july_or_low_n": lambda rows: (
            (rows["game_month"].to_numpy() <= 7)
            | (rows["pitcher_season_n"].to_numpy() <= 200)
        ).astype(float),
    }
    reports = []
    for gate_name, gate_fn in gates.items():
        curves = {}
        for year, item in years.items():
            gate = gate_fn(item["rows"])
            derivative = item["base"] * (1.0 - item["base"])
            first = derivative * item["no_month"]
            second = derivative * gate * item["command"]
            curves[str(year)] = {
                name: curve(item["target"], item["base"], first, second, mask)
                for name, mask in item["masks"].items()
            }
        for no_month_weight in np.arange(.25, 1.251, .05):
            for command_weight in np.arange(.10, 1.251, .05):
                gains = {}
                segment_values = []
                for year in ("2023", "2024"):
                    gains[year] = {}
                    for name, values in curves[year].items():
                        gain = curve_gain(values, no_month_weight, command_weight)
                        gains[year][name] = gain
                        if name != "all":
                            segment_values.append(gain)
                reports.append({
                    "gate": gate_name, "no_month_weight": float(no_month_weight),
                    "command_weight": float(command_weight), "gains": gains,
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "min_segment": min(segment_values),
                    "mean_year": (gains["2023"]["all"] + gains["2024"]["all"]) / 2,
                })
    reports.sort(
        key=lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    # Re-score the best approximate candidates through the exact sigmoid.
    exact_reports = []
    for report in reports[:80]:
        gate_fn = gates[report["gate"]]
        gains = {}
        segment_values = []
        for year, item in years.items():
            gate = gate_fn(item["rows"])
            candidate = sigmoid(
                logit(item["base"])
                + report["no_month_weight"] * item["no_month"]
                + report["command_weight"] * gate * item["command"]
            )
            gains[str(year)] = {}
            for name, mask in item["masks"].items():
                gain = (
                    bss(item["target"][mask], candidate[mask])
                    - bss(item["target"][mask], item["base"][mask])
                )
                gains[str(year)][name] = gain
                if name != "all":
                    segment_values.append(gain)
        exact_reports.append({
            **{key: report[key] for key in ("gate", "no_month_weight", "command_weight")},
            "gains": gains,
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "min_segment": min(segment_values),
            "mean_year": (gains["2023"]["all"] + gains["2024"]["all"]) / 2,
        })
    exact_reports.sort(
        key=lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research" / "v23_joint_command_candidate.json"
    output.write_text(json.dumps({
        "reports": exact_reports, "approximate_reports": reports,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"top": exact_reports[:80]}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
