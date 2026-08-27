"""Validate the deployable R-command and F-resolution portfolio together.

Every correction uses only training-side fitted models/statistics and row-local
features.  In particular, this script never recenters predictions using a
validation/test batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


COMMAND_WEIGHTS = (1.05, 1.00, .80)
FUTURES_POLICIES = {
    "count_100": (.10, 0.0, 0.0),
    "count_125": (.125, 0.0, 0.0),
    "count_150": (.15, 0.0, 0.0),
    "count_175": (.175, 0.0, 0.0),
    "count_200": (.20, 0.0, 0.0),
    "runners_125": (0.0, 0.0, .125),
    "runners_150": (0.0, 0.0, .15),
    "runners_175": (0.0, 0.0, .175),
    "count150_runners050": (.15, 0.0, .05),
    "count200_runners_m050": (.20, 0.0, -.05),
    "count250_runners_m075": (.25, 0.0, -.075),
    "count300_runners_m100": (.30, 0.0, -.10),
}


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


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
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv",
        usecols=[
            "season", "game_month", "game_type", "pitcher_id", "asof_pitcher_n",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    raw = add_pitcher_season_exposure(raw)
    reports = {name: {} for name in FUTURES_POLICIES}
    for year in (2023, 2024):
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        masks = segment_masks(rows)
        with np.load(root / f"research/v23_trackman_no_month_{year}.npz") as z:
            target = z["target"].astype(float)
            base = z["base"].astype(float)
            no_month = z["direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}.npz") as z:
            full_command = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}_w1.npz") as z:
            recent_command = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_conditional_resolution_{year}.npz") as z:
            names = z["names"].astype(str).tolist()
            resolution = {
                name: z["directions"][:, index].astype(float)
                for index, name in enumerate(names)
            }
        if len(rows) != len(target):
            raise ValueError(f"row alignment failed for {year}")

        early_pitcher = rows["pitcher_season_n"].le(600).to_numpy(float)
        regular = masks["regular"].astype(float)
        command_candidate = sigmoid(
            logit(base)
            + regular * (
                COMMAND_WEIGHTS[0] * no_month
                + early_pitcher * COMMAND_WEIGHTS[1] * full_command
                + early_pitcher * COMMAND_WEIGHTS[2] * recent_command
            )
        )
        for policy_name, weights in FUTURES_POLICIES.items():
            correction = sum(
                weight * resolution[name]
                for weight, name in zip(weights, names)
            )
            candidate = np.clip(
                command_candidate + masks["futures"] * correction, .005, .995,
            )
            gains = {
                name: bss(target[mask], candidate[mask]) - bss(
                    target[mask], base[mask],
                )
                for name, mask in masks.items() if mask.any()
            }
            reports[policy_name][str(year)] = gains

    ranked = []
    for policy_name, gains in reports.items():
        temporal = [
            gains[str(year)][name]
            for year in (2023, 2024)
            for name in (
                "first_half", "second_half", "months_3_5", "months_6_7",
                "months_8_11",
            )
        ]
        ranked.append({
            "policy": policy_name,
            "weights": FUTURES_POLICIES[policy_name],
            "gains": gains,
            "min_temporal": min(temporal),
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "mean_year": np.mean([gains["2023"]["all"], gains["2024"]["all"]]),
        })
    ranked.sort(
        key=lambda row: (row["min_temporal"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research/v23_combined_candidate.json"
    output.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print(json.dumps(ranked, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
