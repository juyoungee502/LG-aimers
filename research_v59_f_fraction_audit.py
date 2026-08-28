"""Cross-year robustness audit for the paired F-only fraction specialist."""
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


def cohorts(raw, rows, recent, year):
    futures = rows["game_type"].astype(str).eq("F").to_numpy()
    monotone = recent["recent_fraction_n_monotone"].to_numpy(bool)
    high = pitcher_season_exposure(raw, rows, year) > 100.
    result = {
        "F": futures, "F_monotone": futures & monotone,
        "F_high": futures & high, "F_monotone_high": futures & monotone & high,
    }
    for threshold in (15., 30., 50.):
        enough = recent["recent1_reduced_n"].to_numpy(float) >= threshold
        label = int(threshold)
        result[f"F_n1_{label}"] = futures & enough
        result[f"F_n1_{label}_high"] = futures & enough & high
        result[f"F_monotone_n1_{label}"] = futures & monotone & enough
        result[f"F_monotone_n1_{label}_high"] = futures & monotone & enough & high
    return result


def coefficients(target, base, direction, segments, selected):
    residual = target - base
    correction = direction * selected
    result = {}
    for name, active in segments.items():
        if int(active.sum()) < 500:
            continue
        rate = float(target[active].mean())
        denominator = rate * (1. - rate)
        local = correction[active]
        result[name] = (
            200000. * float(np.mean(residual[active] * local)) / denominator,
            100000. * float(np.mean(local * local)) / denominator,
        )
    return result


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
    years = {}
    for year in (2023, 2024):
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        recent = recent_window_features(rows)
        with np.load(ROOT / "research" / f"v59_f_fraction_s3_{year}.npz") as archive:
            target = archive["target"].astype(float)
            base = np.clip(archive["base"].astype(float), .005, .995)
            valid_f = archive["valid_f"].astype(bool)
            reference = np.clip(archive["reference"].astype(float), .005, .995)
            prediction = np.clip(archive["prediction"].astype(float), .005, .995)
        if len(rows) != len(target) or int(valid_f.sum()) != len(reference):
            raise ValueError(f"row alignment failed for {year}")
        direction = np.zeros(len(rows), dtype=float)
        direction[valid_f] = prediction - reference
        segments = {
            **masks(len(rows)),
            "R": ~valid_f, "F": valid_f,
            **roster_masks(raw, rows, year),
        }
        years[year] = {
            "rows": rows, "target": target, "base": base,
            "direction": direction, "segments": segments,
            "cohorts": cohorts(raw, rows, recent, year),
        }

    names = sorted(set.intersection(*[
        set(values["cohorts"]) for values in years.values()
    ]))
    table = []
    for name in names:
        by_year = {
            year: coefficients(
                values["target"], values["base"], values["direction"],
                values["segments"], values["cohorts"][name],
            )
            for year, values in years.items()
        }
        for weight in np.round(np.arange(-2., 2.0001, .025), 4):
            if abs(weight) < 1e-8:
                continue
            gains = {
                str(year): {
                    segment: linear * weight - quadratic * weight * weight
                    for segment, (linear, quadratic) in by_year[year].items()
                }
                for year in years
            }
            years_all = [gains[str(year)]["all"] for year in years]
            quarters = [
                gains[str(year)][f"q{quarter}"]
                for year in years for quarter in range(1, 5)
            ]
            affected_roster = [
                gains[str(year)][segment]
                for year in years
                for segment in (
                    "F_returning_both", "F_roster_change", "F_same_teams",
                    "F_player_or_team_change", "F_low_pitcher_exposure",
                    "F_high_pitcher_exposure",
                )
                if segment in gains[str(year)]
                and abs(gains[str(year)][segment]) > 1e-12
            ]
            table.append({
                "cohort": name, "weight": float(weight),
                "rows": {
                    str(year): int(values["cohorts"][name].sum())
                    for year, values in years.items()
                },
                "gains": gains,
                "minimum_year_gain": float(min(years_all)),
                "mean_year_gain": float(np.mean(years_all)),
                "minimum_quarter_gain": float(min(quarters)),
                "minimum_affected_roster_gain": float(min(affected_roster)),
            })
    ranked = sorted(
        table,
        key=lambda row: (
            min(row["minimum_year_gain"], row["minimum_quarter_gain"],
                row["minimum_affected_roster_gain"]),
            row["mean_year_gain"],
        ), reverse=True,
    )
    finalists = []
    for row in ranked[:30]:
        year_details = {}
        for year, values in years.items():
            selected = values["cohorts"][row["cohort"]]
            candidate = np.clip(
                values["base"] + row["weight"] * values["direction"] * selected,
                .005, .995,
            )
            team_gains = {}
            for team, indices in values["rows"].groupby(
                "pitcher_team_id", observed=True,
            ).groups.items():
                active = np.zeros(len(candidate), dtype=bool)
                active[np.asarray(list(indices), dtype=int)] = True
                candidate_score = score(values["target"], candidate, active)
                base_score = score(values["target"], values["base"], active)
                if candidate_score is not None:
                    team_gains[str(team)] = float(candidate_score - base_score)
            year_details[str(year)] = {
                "minimum_team_gain": float(min(team_gains.values())),
                "median_team_gain": float(np.median(list(team_gains.values()))),
                "negative_teams": int(sum(value < 0. for value in team_gains.values())),
                "clustered_bootstrap": clustered_interval(
                    values["target"], values["base"], candidate,
                    values["rows"]["pitcher_id"].to_numpy(),
                ),
            }
        finalists.append({**row, "stability": year_details})
    report = {
        "comparison": "paired F-only fraction model minus identical F-only base",
        "best_robust": finalists,
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v59_f_fraction_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
