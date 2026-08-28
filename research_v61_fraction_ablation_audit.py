"""Compare v61 fraction feature groups across available seed ensembles."""
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
VARIANTS = ("full", "confidence", "counts", "shrinkage", "core", "window1")


def archive_path(variant, suffix, year):
    if variant == "full":
        return ROOT / "research" / f"v59_f_fraction_s3{suffix}_{year}.npz"
    return ROOT / "research" / f"v61_fraction_{variant}_s3{suffix}_{year}.npz"


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
    for variant in VARIANTS:
        reports[variant] = {}
        for group, suffix in (("seeds_0_2", ""), ("seeds_3_5", "_o3")):
            paths = [archive_path(variant, suffix, year) for year in (2023, 2024)]
            if not all(path.exists() for path in paths):
                continue
            reports[variant][group] = {}
            for year, path in zip((2023, 2024), paths):
                rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
                recent = recent_window_features(rows)
                with np.load(path) as archive:
                    target = archive["target"].astype(float)
                    base = np.clip(archive["base"].astype(float), .005, .995)
                    valid_f = archive["valid_f"].astype(bool)
                    reference = archive["reference"].astype(float)
                    prediction = archive["prediction"].astype(float)
                selected = (
                    valid_f
                    & (recent["recent1_reduced_n"].to_numpy(float) >= 30.)
                    & (pitcher_season_exposure(raw, rows, year) > 100.)
                )
                direction = np.zeros(len(rows), dtype=float)
                direction[valid_f] = prediction - reference
                segments = {**masks(len(rows)), **roster_masks(raw, rows, year)}
                baseline = {
                    name: score(target, base, active) for name, active in segments.items()
                }
                reports[variant][group][str(year)] = {
                    "target": target, "base": base, "direction": direction,
                    "selected": selected, "segments": segments,
                    "baseline": baseline, "rows": rows,
                }

    summary = {}
    details = {}
    for variant, groups in reports.items():
        if not groups:
            continue
        for weight in (.25, .50, .75, 1.00):
            name = f"{variant}_{weight:.2f}"
            items = []
            details[name] = {}
            for group, years in groups.items():
                details[name][group] = {}
                for year, values in years.items():
                    candidate = np.clip(
                        values["base"]
                        + weight * values["direction"] * values["selected"],
                        .005, .995,
                    )
                    gains = {}
                    for segment, active in values["segments"].items():
                        candidate_score = score(values["target"], candidate, active)
                        if candidate_score is not None:
                            gains[segment] = candidate_score - values["baseline"][segment]
                    team_gains = {}
                    for team, indices in values["rows"].groupby(
                        "pitcher_team_id", observed=True,
                    ).groups.items():
                        active = np.zeros(len(candidate), bool)
                        active[np.asarray(list(indices), dtype=int)] = True
                        candidate_score = score(values["target"], candidate, active)
                        base_score = score(values["target"], values["base"], active)
                        if candidate_score is not None:
                            team_gains[str(team)] = candidate_score - base_score
                    item = {
                        "gain_all": float(gains["all"]),
                        "minimum_quarter_gain": float(min(gains[f"q{i}"] for i in range(1, 5))),
                        "minimum_affected_roster_gain": float(min(
                            gains[key] for key in (
                                "F_returning_both", "F_roster_change", "F_same_teams",
                                "F_player_or_team_change", "F_high_pitcher_exposure",
                            ) if key in gains
                        )),
                        "minimum_team_gain": float(min(team_gains.values())),
                        "negative_teams": int(sum(value < 0. for value in team_gains.values())),
                        "clustered_bootstrap": clustered_interval(
                            values["target"], values["base"], candidate,
                            values["rows"]["pitcher_id"].to_numpy(), repeats=2000,
                        ),
                    }
                    details[name][group][year] = item
                    items.append(item)
            summary[name] = {
                "groups": len(groups),
                "minimum_all_gain": float(min(item["gain_all"] for item in items)),
                "mean_all_gain": float(np.mean([item["gain_all"] for item in items])),
                "minimum_quarter_gain": float(min(item["minimum_quarter_gain"] for item in items)),
                "minimum_affected_roster_gain": float(min(
                    item["minimum_affected_roster_gain"] for item in items
                )),
                "minimum_bootstrap_p05": float(min(
                    item["clustered_bootstrap"]["p05"] for item in items
                )),
                "minimum_positive_probability": float(min(
                    item["clustered_bootstrap"]["positive_probability"] for item in items
                )),
                "minimum_team_gain": float(min(item["minimum_team_gain"] for item in items)),
                "maximum_negative_teams": int(max(item["negative_teams"] for item in items)),
            }
    ranked = sorted(
        summary,
        key=lambda name: (
            summary[name]["minimum_quarter_gain"],
            summary[name]["minimum_all_gain"],
            summary[name]["mean_all_gain"],
        ), reverse=True,
    )
    report = {
        "summary": {name: summary[name] for name in ranked},
        "details": details,
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v61_fraction_ablation_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
