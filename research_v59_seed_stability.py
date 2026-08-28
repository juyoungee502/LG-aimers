"""Audit the F-only fraction correction independently for every paired seed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recent_window_features import recent_window_features
from research_v40_failure_seed_stability import masks
from research_v58_fraction_roster_stability import (
    clustered_interval, pitcher_season_exposure, roster_masks, score,
)


ROOT = Path(__file__).resolve().parent


def main():
    usecols = [
        "season", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
        "asof_pitcher_success_rate",
        *[
            f"asof_pitcher_prev{window}_game_{label}_rate"
            for window in (1, 3, 5) for label in ("success", "middle")
        ],
    ]
    raw = pd.read_csv(
        ROOT / "data/train.csv", usecols=usecols,
        encoding="utf-8-sig", low_memory=False,
    )
    reports = {}
    for weight in (.50, .75, 1.00):
        key = f"F_n1_30_high_{weight:.2f}"
        reports[key] = {}
        for year in (2023, 2024):
            rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
            recent = recent_window_features(rows)
            with np.load(ROOT / "research" / f"v59_f_fraction_s3_{year}.npz") as z:
                target = z["target"].astype(float)
                base = np.clip(z["base"].astype(float), .005, .995)
                valid_f = z["valid_f"].astype(bool)
                references = z["reference_members"].astype(float)
                predictions = z["prediction_members"].astype(float)
            if references.shape != predictions.shape or references.shape[0] != 3:
                raise ValueError(f"expected three paired members for {year}")
            selected = (
                valid_f
                & (recent["recent1_reduced_n"].to_numpy(float) >= 30.)
                & (pitcher_season_exposure(raw, rows, year) > 100.)
            )
            segments = {**masks(len(rows)), **roster_masks(raw, rows, year)}
            baseline = {
                name: score(target, base, active) for name, active in segments.items()
            }
            seed_reports = []
            for seed_index in range(3):
                direction = np.zeros(len(rows), dtype=float)
                direction[valid_f] = (
                    predictions[seed_index] - references[seed_index]
                )
                candidate = np.clip(base + weight * direction * selected, .005, .995)
                scores = {
                    name: score(target, candidate, active)
                    for name, active in segments.items()
                }
                gains = {
                    name: scores[name] - baseline[name]
                    for name in segments if scores[name] is not None
                }
                team_gains = {}
                for team, indices in rows.groupby(
                    "pitcher_team_id", observed=True,
                ).groups.items():
                    active = np.zeros(len(rows), bool)
                    active[np.asarray(list(indices), dtype=int)] = True
                    candidate_score = score(target, candidate, active)
                    base_score = score(target, base, active)
                    if candidate_score is not None:
                        team_gains[str(team)] = float(candidate_score - base_score)
                seed_reports.append({
                    "seed_index": seed_index,
                    "gains": gains,
                    "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "minimum_affected_roster_gain": float(min(
                        gains[name] for name in (
                            "F_returning_both", "F_roster_change", "F_same_teams",
                            "F_player_or_team_change", "F_high_pitcher_exposure",
                        ) if name in gains
                    )),
                    "minimum_team_gain": float(min(team_gains.values())),
                    "negative_teams": int(sum(value < 0. for value in team_gains.values())),
                    "clustered_bootstrap": clustered_interval(
                        target, base, candidate, rows["pitcher_id"].to_numpy(),
                        repeats=2000,
                    ),
                })
            reports[key][str(year)] = {
                "selected_rows": int(selected.sum()), "seeds": seed_reports,
            }

    summary = {}
    for name, years in reports.items():
        members = [
            seed for year in years.values() for seed in year["seeds"]
        ]
        summary[name] = {
            "minimum_all_gain": float(min(seed["gains"]["all"] for seed in members)),
            "minimum_quarter_gain": float(min(seed["minimum_quarter_gain"] for seed in members)),
            "minimum_affected_roster_gain": float(min(
                seed["minimum_affected_roster_gain"] for seed in members
            )),
            "minimum_bootstrap_p05": float(min(
                seed["clustered_bootstrap"]["p05"] for seed in members
            )),
            "minimum_positive_probability": float(min(
                seed["clustered_bootstrap"]["positive_probability"] for seed in members
            )),
            "minimum_team_gain": float(min(seed["minimum_team_gain"] for seed in members)),
            "maximum_negative_teams": int(max(seed["negative_teams"] for seed in members)),
        }
    output_report = {
        "summary": summary, "reports": reports,
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v59_seed_stability.json"
    output.write_text(json.dumps(output_report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
