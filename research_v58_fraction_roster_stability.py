"""Audit paired recent-fraction corrections under player and team turnover.

The correction is the prediction difference between two otherwise identical
direct CatBoost models.  Only the second model sees row-local features recovered
from the official recent-game rate fractions.  This isolates feature value from
the much weaker standalone direct model and applies it over frozen v54 OOF
predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recent_window_features import recent_window_features
from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import masks


ROOT = Path(__file__).resolve().parent


def score(target, prediction, active):
    if int(active.sum()) < 500:
        return None
    return float(bss(target[active], prediction[active]))


def clustered_interval(target, base, prediction, pitcher_ids, repeats=4000):
    delta = (target - base) ** 2 - (target - prediction) ** 2
    frame = pd.DataFrame({"pitcher": pitcher_ids, "delta": delta})
    groups = frame.groupby("pitcher", observed=True)["delta"].agg(["sum", "count"])
    values = groups[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(5800)
    draws = np.empty(repeats, dtype=float)
    denominator = float(target.mean() * (1. - target.mean()))
    for index in range(repeats):
        selected = rng.integers(0, len(values), size=len(values))
        sampled = values[selected].sum(axis=0)
        draws[index] = 100000. * sampled[0] / sampled[1] / denominator
    return {
        "p05": float(np.quantile(draws, .05)),
        "median": float(np.quantile(draws, .50)),
        "p95": float(np.quantile(draws, .95)),
        "positive_probability": float(np.mean(draws > 0.)),
        "pitcher_clusters": int(len(values)),
    }


def pitcher_season_exposure(raw, rows, year):
    previous = raw.loc[raw["season"].eq(year - 1)]
    previous_pitcher_end = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["asof_pitcher_n"] + 1.
    )
    origin = rows["pitcher_id"].map(previous_pitcher_end).fillna(0.).to_numpy(float)
    return np.maximum(0., rows["asof_pitcher_n"].to_numpy(float) - origin)


def roster_masks(raw, rows, year):
    previous = raw.loc[raw["season"].eq(year - 1)]
    previous_pitchers = set(previous["pitcher_id"].astype(int))
    previous_batters = set(previous["batter_id"].astype(int))
    pitcher_returning = rows["pitcher_id"].astype(int).isin(previous_pitchers).to_numpy()
    batter_returning = rows["batter_id"].astype(int).isin(previous_batters).to_numpy()

    last_pitcher_team = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["pitcher_team_id"]
    )
    last_batter_team = (
        previous.groupby("batter_id", observed=True, sort=False).tail(1)
        .set_index("batter_id")["batter_team_id"]
    )
    prior_pitcher_team = rows["pitcher_id"].map(last_pitcher_team)
    prior_batter_team = rows["batter_id"].map(last_batter_team)
    same_pitcher_team = (
        prior_pitcher_team.eq(rows["pitcher_team_id"]) & pitcher_returning
    ).to_numpy()
    same_batter_team = (
        prior_batter_team.eq(rows["batter_team_id"]) & batter_returning
    ).to_numpy()

    exposure = pitcher_season_exposure(raw, rows, year)
    futures = rows["game_type"].astype(str).eq("F").to_numpy()
    result = {
        "returning_both": pitcher_returning & batter_returning,
        "roster_change": ~(pitcher_returning & batter_returning),
        "same_teams": same_pitcher_team & same_batter_team,
        "player_or_team_change": ~(same_pitcher_team & same_batter_team),
        "low_pitcher_exposure": exposure <= 100.,
        "high_pitcher_exposure": exposure > 100.,
        "F_returning_both": futures & pitcher_returning & batter_returning,
        "F_roster_change": futures & ~(pitcher_returning & batter_returning),
        "F_same_teams": futures & same_pitcher_team & same_batter_team,
        "F_player_or_team_change": futures & ~(same_pitcher_team & same_batter_team),
        "F_low_pitcher_exposure": futures & (exposure <= 100.),
        "F_high_pitcher_exposure": futures & (exposure > 100.),
    }
    return result


def candidate_specs(raw, rows, recent, year):
    game_type = rows["game_type"].astype(str).to_numpy()
    monotone = recent["recent_fraction_n_monotone"].to_numpy(bool)
    f_n1_50 = (
        (game_type == "F")
        & (recent["recent1_reduced_n"].to_numpy(float) >= 50.)
    )
    high_exposure = pitcher_season_exposure(raw, rows, year) > 100.
    monotone_high = monotone & high_exposure
    f_n1_50_high = f_n1_50 & high_exposure
    # Broad conservative correction, high-confidence F correction, and their
    # combination.  We retain several scales so the roster audit, not a single
    # aggregate score, chooses the production setting.
    specs = {
        "monotone_0.15": [(monotone, .15)],
        "monotone_0.20": [(monotone, .20)],
        "monotone_0.25": [(monotone, .25)],
    }
    for weight in (.50, 1.00, 1.50, 2.00):
        specs[f"F_n1_50_{weight:.2f}"] = [(f_n1_50, weight)]
        specs[f"F_n1_50_high_{weight:.2f}"] = [(f_n1_50_high, weight)]
    for weight in (.15, .20, .25):
        specs[f"monotone_high_{weight:.2f}"] = [(monotone_high, weight)]
    for broad, focused in ((.15, 1.00), (.15, 1.50), (.20, 1.00), (.20, 1.50)):
        specs[f"monotone_{broad:.2f}_plus_F_n1_50_{focused:.2f}"] = [
            (monotone, broad), (f_n1_50, focused),
        ]
        specs[f"monotone_high_{broad:.2f}_plus_F_n1_50_high_{focused:.2f}"] = [
            (monotone_high, broad), (f_n1_50_high, focused),
        ]
    return specs


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
    yearly = {}
    for year in (2023, 2024):
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        recent = recent_window_features(rows)
        with np.load(
            ROOT / "research" / f"v58_recent_fraction_hl2_s3_{year}.npz"
        ) as archive:
            target = archive["target"].astype(float)
            base = np.clip(archive["base"].astype(float), .005, .995)
            extra = np.clip(archive["prediction"].astype(float), .005, .995)
        with np.load(
            ROOT / "research" / f"v35_lowcard_direct_hl2_s3_{year}.npz"
        ) as archive:
            reference = np.clip(archive["prediction"].astype(float), .005, .995)
        if not len(rows) == len(target) == len(base) == len(extra) == len(reference):
            raise ValueError(f"row alignment failed for {year}")
        direction = extra - reference
        game_type = rows["game_type"].astype(str).to_numpy()
        cohorts = {
            "all": np.ones(len(rows), bool),
            "R": game_type == "R", "F": game_type == "F",
            **masks(len(rows)), **roster_masks(raw, rows, year),
        }
        baseline = {
            name: score(target, base, active) for name, active in cohorts.items()
        }
        reports = {}
        for name, pieces in candidate_specs(raw, rows, recent, year).items():
            correction = np.zeros(len(rows), dtype=float)
            selected = np.zeros(len(rows), dtype=bool)
            for active, weight in pieces:
                correction[active] += weight * direction[active]
                selected |= active
            prediction = np.clip(base + correction, .005, .995)
            scores = {
                cohort: score(target, prediction, active)
                for cohort, active in cohorts.items()
            }
            gains = {
                cohort: scores[cohort] - baseline[cohort]
                for cohort in cohorts if scores[cohort] is not None
            }
            team_gains = {}
            for team, indices in rows.groupby(
                "pitcher_team_id", observed=True,
            ).groups.items():
                active = np.zeros(len(rows), dtype=bool)
                active[np.asarray(list(indices), dtype=int)] = True
                value = score(target, prediction, active)
                reference_score = score(target, base, active)
                if value is not None:
                    team_gains[str(team)] = float(value - reference_score)
            reports[name] = {
                "selected_rows": int(selected.sum()),
                "gains": gains,
                "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "minimum_roster_gain": float(min(
                    gains[key] for key in (
                        "returning_both", "roster_change", "same_teams",
                        "player_or_team_change", "low_pitcher_exposure",
                        "high_pitcher_exposure",
                    ) if key in gains
                )),
                "minimum_F_roster_gain": float(min(
                    gains[key] for key in (
                        "F_returning_both", "F_roster_change", "F_same_teams",
                        "F_player_or_team_change", "F_low_pitcher_exposure",
                        "F_high_pitcher_exposure",
                    ) if key in gains
                )),
                "minimum_team_gain": float(min(team_gains.values())),
                "median_team_gain": float(np.median(list(team_gains.values()))),
                "negative_teams": int(sum(value < 0. for value in team_gains.values())),
                "team_gains": team_gains,
                "clustered_bootstrap": clustered_interval(
                    target, base, prediction, rows["pitcher_id"].to_numpy(),
                ),
            }
        yearly[str(year)] = {
            "cohort_rows": {name: int(active.sum()) for name, active in cohorts.items()},
            "reports": reports,
        }

    names = sorted(set.intersection(*[
        set(value["reports"]) for value in yearly.values()
    ]))
    summary = {}
    for name in names:
        reports = [yearly[str(year)]["reports"][name] for year in (2023, 2024)]
        summary[name] = {
            "gains": {str(year): yearly[str(year)]["reports"][name]["gains"]["all"]
                      for year in (2023, 2024)},
            "minimum_year_gain": float(min(report["gains"]["all"] for report in reports)),
            "minimum_quarter_gain": float(min(report["minimum_quarter_gain"] for report in reports)),
            "minimum_roster_gain": float(min(report["minimum_roster_gain"] for report in reports)),
            "minimum_F_roster_gain": float(min(report["minimum_F_roster_gain"] for report in reports)),
            "minimum_team_gain": float(min(report["minimum_team_gain"] for report in reports)),
            "maximum_negative_teams": int(max(report["negative_teams"] for report in reports)),
            "minimum_bootstrap_p05": float(min(
                report["clustered_bootstrap"]["p05"] for report in reports
            )),
            "minimum_positive_probability": float(min(
                report["clustered_bootstrap"]["positive_probability"] for report in reports
            )),
        }
    ranked = sorted(
        summary,
        key=lambda name: (
            summary[name]["minimum_bootstrap_p05"] > 0.,
            summary[name]["minimum_roster_gain"],
            summary[name]["minimum_quarter_gain"],
            summary[name]["minimum_year_gain"],
        ), reverse=True,
    )
    report = {
        "method": "paired v58 minus v35 probability correction over frozen v54",
        "summary": {name: summary[name] for name in ranked},
        "yearly": yearly,
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v58_fraction_roster_stability.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
