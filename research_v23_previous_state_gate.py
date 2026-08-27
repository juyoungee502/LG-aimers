"""Screen row-local expiry gates for the previous-season state direction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


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


def masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv",
        usecols=[
            "season", "game_month", "game_type", "pitcher_id", "asof_pitcher_n",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    data = add_pitcher_season_exposure(data)
    years = {}
    for year in (2023, 2024):
        with np.load(root / f"research/v23_previous_season_state_{year}.npz") as z:
            target = z["target"].astype(float)
            base = z["base"].astype(float)
            direction = z["direction"].astype(float)
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        years[year] = (target, base, direction, rows, masks(rows))

    gate_functions = {
        "month_5": lambda rows: rows["game_month"].le(5).to_numpy(float),
        "month_7": lambda rows: rows["game_month"].le(7).to_numpy(float),
        "pitch_n_100": lambda rows: rows["pitcher_season_n"].le(100).to_numpy(float),
        "pitch_n_200": lambda rows: rows["pitcher_season_n"].le(200).to_numpy(float),
        "pitch_n_400": lambda rows: rows["pitcher_season_n"].le(400).to_numpy(float),
        "pitch_n_600": lambda rows: rows["pitcher_season_n"].le(600).to_numpy(float),
        "pitch_decay_100": lambda rows: 100.0 / (100.0 + rows["pitcher_season_n"].to_numpy()),
        "pitch_decay_300": lambda rows: 300.0 / (300.0 + rows["pitcher_season_n"].to_numpy()),
    }
    reports = []
    for gate_name, gate_function in gate_functions.items():
        for regime in ("all", "regular", "futures"):
            for weight in np.arange(.025, .501, .025):
                gains = {}
                temporal = []
                for year, (target, base, direction, rows, year_masks) in years.items():
                    gate = gate_function(rows)
                    if regime != "all":
                        gate *= rows["game_type"].eq("R" if regime == "regular" else "F").to_numpy()
                    candidate = np.clip(base + weight * gate * direction, .005, .995)
                    gains[str(year)] = {
                        name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
                        for name, mask in year_masks.items()
                    }
                    temporal.extend(
                        gains[str(year)][name]
                        for name in (
                            "first_half", "second_half", "months_3_5", "months_6_7",
                            "months_8_11",
                        )
                    )
                reports.append({
                    "gate": gate_name, "regime": regime, "weight": float(weight),
                    "gains": gains, "min_temporal": min(temporal),
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "mean_year": np.mean([gains["2023"]["all"], gains["2024"]["all"]]),
                })
    reports.sort(
        key=lambda row: (row["min_temporal"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research/v23_previous_state_gate.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"top": reports[:80]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
