"""Audit regime-specific exposure gates for the paired v66 auxiliary signal.

The v66 ungated audit showed an interpretable interaction fixed before this
screen: R benefited among established/high-exposure pitchers, while F benefited
among low-exposure pitchers.  This script tests only that small policy family
with one configuration shared across 2023 and 2024.  A later independent seed
replication is mandatory before any production promotion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import logit, sigmoid
from research_v58_fraction_roster_stability import (
    clustered_interval, pitcher_season_exposure, roster_masks,
)


ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024)
THRESHOLDS = (50., 100., 150., 200., 300.)
R_WEIGHTS = (-.10, -.20, -.30)
F_WEIGHTS = (.10, .20, .30)
R_POLICIES = ("R_high", "R_changed_high", "R_returning_high")
F_POLICIES = ("F_low", "F_changed_low", "F_returning_low")


def gain_from_delta(target, delta, mask):
    if int(mask.sum()) < 500:
        return None
    rate = float(target[mask].mean())
    return float(100000. * delta[mask].mean() / (rate * (1. - rate)))


def time_masks(length):
    position = np.arange(length)
    boundaries = np.linspace(0, length, 5, dtype=int)
    result = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index in range(4):
        result[f"q{index + 1}"] = (
            (position >= boundaries[index]) & (position < boundaries[index + 1])
        )
    return result


def selectors(raw, rows, year, threshold):
    exposure = pitcher_season_exposure(raw, rows, year)
    roster = roster_masks(raw, rows, year)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    futures = ~regular
    high = exposure > threshold
    low = ~high
    return {
        "R_high": regular & high,
        "R_changed_high": regular & high & roster["player_or_team_change"],
        "R_returning_high": regular & high & roster["returning_both"],
        "F_low": futures & low,
        "F_changed_low": futures & low & roster["player_or_team_change"],
        "F_returning_low": futures & low & roster["returning_both"],
    }, exposure, roster


def metrics(target, base, prediction, masks, selected_r, selected_f):
    delta = (target - base) ** 2 - (target - prediction) ** 2
    gains = {}
    for name, mask in masks.items():
        gain = gain_from_delta(target, delta, mask)
        if gain is not None:
            gains[name] = gain
    selected = {}
    for name, mask in (("selected_R", selected_r), ("selected_F", selected_f)):
        gain = gain_from_delta(target, delta, mask)
        if gain is not None:
            selected[name] = gain
    chronology = {name: gains[name] for name in ("h1", "h2", "q1", "q2", "q3", "q4")}
    cohorts = {
        name: value for name, value in gains.items()
        if name.startswith("roster_")
        and not name.endswith(("low_pitcher_exposure", "high_pitcher_exposure"))
    }
    affected_cohorts = {name: value for name, value in cohorts.items() if abs(value) > 1e-12}
    return {
        "gain_all": gains["all"],
        "gain_R": gains["R"],
        "gain_F": gains["F"],
        "chronological": chronology,
        "selected": selected,
        "affected_roster": affected_cohorts,
    }


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}

    data = {}
    for year in YEARS:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        active = oof["season"] == year
        target = oof["target"][active].astype(float)
        base = np.clip(oof["blended"][active].astype(float), .005, .995)
        if not np.array_equal(rows["control_success"].to_numpy(float), target):
            raise ValueError(f"v54 rows do not align for {year}")
        with np.load(ROOT / f"research/v66_multitask_tabm_{year}.npz") as archive:
            directions = {
                epoch: (
                    logit(np.clip(archive[f"aux_epoch_{epoch}"].astype(float), .005, .995))
                    - logit(np.clip(archive[f"main_epoch_{epoch}"].astype(float), .005, .995))
                )
                for epoch in range(1, 5)
            }
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        base_masks = {**time_masks(len(rows)), "R": regular, "F": ~regular}
        data[year] = {
            "rows": rows, "target": target, "base": base,
            "directions": directions, "base_masks": base_masks,
        }

    selector_cache = {
        year: {
            threshold: selectors(raw, data[year]["rows"], year, threshold)
            for threshold in THRESHOLDS
        }
        for year in YEARS
    }

    configurations = []
    for epoch in range(1, 5):
        for threshold in THRESHOLDS:
            for policy in R_POLICIES:
                for weight in R_WEIGHTS:
                    configurations.append({
                        "epoch": epoch, "threshold": threshold,
                        "r_policy": policy, "r_weight": weight,
                        "f_policy": None, "f_weight": 0.,
                    })
            for policy in F_POLICIES:
                for weight in F_WEIGHTS:
                    configurations.append({
                        "epoch": epoch, "threshold": threshold,
                        "r_policy": None, "r_weight": 0.,
                        "f_policy": policy, "f_weight": weight,
                    })
            for r_weight in R_WEIGHTS:
                for f_weight in F_WEIGHTS:
                    configurations.append({
                        "epoch": epoch, "threshold": threshold,
                        "r_policy": "R_high", "r_weight": r_weight,
                        "f_policy": "F_low", "f_weight": f_weight,
                    })

    reports = []
    for configuration in configurations:
        years = {}
        chronology_values = []
        cohort_values = []
        selected_values = []
        predictions = {}
        for year in YEARS:
            block = data[year]
            selected, exposure, roster = selector_cache[year][configuration["threshold"]]
            selected_r = (
                selected[configuration["r_policy"]]
                if configuration["r_policy"] else np.zeros(len(exposure), dtype=bool)
            )
            selected_f = (
                selected[configuration["f_policy"]]
                if configuration["f_policy"] else np.zeros(len(exposure), dtype=bool)
            )
            correction = np.zeros(len(exposure), dtype=float)
            direction = block["directions"][configuration["epoch"]]
            correction[selected_r] = configuration["r_weight"] * direction[selected_r]
            correction[selected_f] = configuration["f_weight"] * direction[selected_f]
            prediction = sigmoid(logit(block["base"]) + correction)
            masks = {
                **block["base_masks"],
                **{f"roster_{name}": mask for name, mask in roster.items()},
            }
            result = metrics(
                block["target"], block["base"], prediction, masks,
                selected_r, selected_f,
            )
            result["selected_R_rows"] = int(selected_r.sum())
            result["selected_F_rows"] = int(selected_f.sum())
            years[str(year)] = result
            chronology_values.extend(result["chronological"].values())
            cohort_values.extend(result["affected_roster"].values())
            selected_values.extend(result["selected"].values())
            predictions[year] = prediction
        reports.append({
            **configuration,
            "years": years,
            "minimum_year_gain": float(min(years[str(year)]["gain_all"] for year in YEARS)),
            "mean_year_gain": float(np.mean([years[str(year)]["gain_all"] for year in YEARS])),
            "minimum_chronological_gain": float(min(chronology_values)),
            "minimum_affected_roster_gain": float(min(cohort_values)) if cohort_values else 0.,
            "minimum_selected_gain": float(min(selected_values)) if selected_values else 0.,
        })

    ranked = sorted(reports, key=lambda row: (
        row["minimum_year_gain"], row["minimum_chronological_gain"],
        row["minimum_affected_roster_gain"], row["mean_year_gain"],
    ), reverse=True)
    audited = ranked[:50]
    for report in audited:
        intervals = {}
        for year in YEARS:
            block = data[year]
            selected, _, _ = selector_cache[year][report["threshold"]]
            correction = np.zeros(len(block["target"]), dtype=float)
            direction = block["directions"][report["epoch"]]
            if report["r_policy"]:
                active = selected[report["r_policy"]]
                correction[active] = report["r_weight"] * direction[active]
            if report["f_policy"]:
                active = selected[report["f_policy"]]
                correction[active] = report["f_weight"] * direction[active]
            prediction = sigmoid(logit(block["base"]) + correction)
            intervals[str(year)] = clustered_interval(
                block["target"], block["base"], prediction,
                block["rows"]["pitcher_id"].to_numpy(), repeats=2000,
            )
        report["pitcher_clustered_bootstrap"] = intervals

    safe = sorted([
        row for row in audited
        if row["minimum_year_gain"] > 0.
        and row["minimum_chronological_gain"] > 0.
        and row["minimum_affected_roster_gain"] > 0.
        and row["minimum_selected_gain"] > 0.
        and min(
            row["pitcher_clustered_bootstrap"][str(year)]["p05"]
            for year in YEARS
        ) > 0.
    ], key=lambda row: row["mean_year_gain"], reverse=True)
    output = {
        "anchor": {"version": "v54", "public_score": 1113},
        "hypothesis": (
            "Use the paired auxiliary direction only for high-exposure R or "
            "low-exposure F pitchers; interaction chosen from the ungated v66 audit."
        ),
        "independent_seed_replication_required": True,
        "safe": safe,
        "ranked": ranked[:80],
        "row_independent": True,
        "current_pitch_information_used_at_inference": False,
        "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v67_multitask_exposure_gate.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe[:15], "top": ranked[:15]}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
