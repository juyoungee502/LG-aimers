"""Audit six-seed portfolios of the full and recent-one-game corrections."""
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
FULL_WEIGHTS = (0., .25, .50, .75, 1.00)
WINDOW1_WEIGHTS = (0., .25, .50, .75)


def load_direction(variant: str, suffix: str, year: int):
    stem = "v59_f_fraction" if variant == "full" else "v61_fraction_window1"
    path = ROOT / "research" / f"{stem}_s3{suffix}_{year}.npz"
    with np.load(path) as archive:
        target = archive["target"].astype(float)
        base = np.clip(archive["base"].astype(float), .005, .995)
        valid_f = archive["valid_f"].astype(bool)
        direction = np.zeros(len(base), dtype=float)
        direction[valid_f] = (
            archive["prediction"].astype(float)
            - archive["reference"].astype(float)
        )
    return target, base, valid_f, direction


def evaluate(target, base, candidate, rows, segments):
    gains = {}
    for name, active in segments.items():
        candidate_score = score(target, candidate, active)
        base_score = score(target, base, active)
        if candidate_score is not None and base_score is not None:
            gains[name] = candidate_score - base_score
    team_gains = {}
    for team, indices in rows.groupby("pitcher_team_id", observed=True).groups.items():
        active = np.zeros(len(rows), dtype=bool)
        active[np.asarray(list(indices), dtype=int)] = True
        team_gains[str(team)] = (
            score(target, candidate, active) - score(target, base, active)
        )
    return {
        "gain_all": float(gains["all"]),
        "quarter_gains": {f"q{i}": float(gains[f"q{i}"]) for i in range(1, 5)},
        "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
        "affected_roster_gains": {
            key: float(gains[key]) for key in (
                "F_returning_both", "F_roster_change", "F_same_teams",
                "F_player_or_team_change", "F_high_pitcher_exposure",
            ) if key in gains
        },
        "minimum_affected_roster_gain": float(min(
            gains[key] for key in (
                "F_returning_both", "F_roster_change", "F_same_teams",
                "F_player_or_team_change", "F_high_pitcher_exposure",
            ) if key in gains
        )),
        "minimum_team_gain": float(min(team_gains.values())),
        "negative_teams": int(sum(value < 0. for value in team_gains.values())),
        "clustered_bootstrap": clustered_interval(
            target, base, candidate, rows["pitcher_id"].to_numpy(), repeats=3000,
        ),
    }


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
        payloads = {}
        for variant in ("full", "window1"):
            payloads[variant] = []
            for suffix in ("", "_o3"):
                payload = load_direction(variant, suffix, year)
                payloads[variant].append(payload)
        target, base, valid_f, _ = payloads["full"][0]
        for variant_payloads in payloads.values():
            for other_target, other_base, other_valid_f, _ in variant_payloads:
                if not (
                    np.array_equal(target, other_target)
                    and np.array_equal(base, other_base)
                    and np.array_equal(valid_f, other_valid_f)
                ):
                    raise ValueError(f"Archive mismatch for {year}")
        selected = (
            valid_f
            & (recent["recent1_reduced_n"].to_numpy(float) >= 30.)
            & (pitcher_season_exposure(raw, rows, year) > 100.)
        )
        years[year] = {
            "target": target,
            "base": base,
            "rows": rows,
            "selected": selected,
            "segments": {**masks(len(rows)), **roster_masks(raw, rows, year)},
            "directions": {
                variant: [payload[3] for payload in variant_payloads]
                for variant, variant_payloads in payloads.items()
            },
        }

    reports = {}
    for full_weight in FULL_WEIGHTS:
        for window1_weight in WINDOW1_WEIGHTS:
            if full_weight == window1_weight == 0.:
                continue
            name = f"full_{full_weight:.2f}_window1_{window1_weight:.2f}"
            production = {}
            group_stress = {"seeds_0_2": {}, "seeds_3_5": {}}
            for year, values in years.items():
                full_groups = values["directions"]["full"]
                window1_groups = values["directions"]["window1"]
                direction = (
                    full_weight * np.mean(full_groups, axis=0)
                    + window1_weight * np.mean(window1_groups, axis=0)
                )
                candidate = np.clip(
                    values["base"] + direction * values["selected"], .005, .995,
                )
                production[str(year)] = evaluate(
                    values["target"], values["base"], candidate,
                    values["rows"], values["segments"],
                )
                for group_index, group_name in enumerate(group_stress):
                    group_direction = (
                        full_weight * full_groups[group_index]
                        + window1_weight * window1_groups[group_index]
                    )
                    group_candidate = np.clip(
                        values["base"]
                        + group_direction * values["selected"], .005, .995,
                    )
                    group_stress[group_name][str(year)] = evaluate(
                        values["target"], values["base"], group_candidate,
                        values["rows"], values["segments"],
                    )
            production_items = list(production.values())
            stress_items = [
                item for group in group_stress.values() for item in group.values()
            ]
            reports[name] = {
                "full_weight": full_weight,
                "window1_weight": window1_weight,
                "summary": {
                    "minimum_year_gain": float(min(
                        item["gain_all"] for item in production_items
                    )),
                    "mean_year_gain": float(np.mean([
                        item["gain_all"] for item in production_items
                    ])),
                    "minimum_quarter_gain": float(min(
                        item["minimum_quarter_gain"] for item in production_items
                    )),
                    "minimum_affected_roster_gain": float(min(
                        item["minimum_affected_roster_gain"] for item in production_items
                    )),
                    "minimum_bootstrap_p05": float(min(
                        item["clustered_bootstrap"]["p05"] for item in production_items
                    )),
                    "minimum_group_year_gain": float(min(
                        item["gain_all"] for item in stress_items
                    )),
                    "minimum_group_quarter_gain": float(min(
                        item["minimum_quarter_gain"] for item in stress_items
                    )),
                    "minimum_group_roster_gain": float(min(
                        item["minimum_affected_roster_gain"] for item in stress_items
                    )),
                },
                "production": production,
                "group_stress": group_stress,
            }
    ranked = sorted(
        reports,
        key=lambda name: (
            reports[name]["summary"]["minimum_group_year_gain"] > 0.,
            reports[name]["summary"]["minimum_group_quarter_gain"] > 0.,
            reports[name]["summary"]["minimum_group_roster_gain"] > 0.,
            reports[name]["summary"]["minimum_year_gain"],
            reports[name]["summary"]["mean_year_gain"],
        ),
        reverse=True,
    )
    report = {
        "ranking": ranked,
        "candidates": {name: reports[name] for name in ranked},
        "selection_policy": (
            "Require positive independent-group year, quarter, and roster stress; "
            "then maximize the weaker production year."
        ),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v62_fraction_portfolio.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        name: reports[name]["summary"] for name in ranked[:12]
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
