"""Isolate the increment from recent-fraction features over the same direct model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recent_window_features import recent_window_features
from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import masks


ROOT = Path(__file__).resolve().parent


def score(target, prediction, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in masks(len(target)).items()
    }
    for regime in ("R", "F"):
        active = game_type == regime
        result[regime] = float(bss(target[active], prediction[active]))
    return result


def gain_coefficients(target, base, direction, game_type, cohort):
    """Exact un-clipped Brier gain coefficients for a linear delta blend."""
    segments = masks(len(target))
    segments.update({
        "R": game_type == "R",
        "F": game_type == "F",
    })
    residual = target - base
    correction = direction * cohort
    result = {}
    for name, active in segments.items():
        rate = float(target[active].mean())
        reference = rate * (1. - rate)
        local = correction[active]
        result[name] = (
            200000. * float(np.mean(residual[active] * local)) / reference,
            100000. * float(np.mean(local * local)) / reference,
        )
    return result


def cohort_masks(frame, recent):
    game_type = frame["game_type"].astype(str).to_numpy()
    regular = game_type == "R"
    futures = game_type == "F"
    monotone = recent["recent_fraction_n_monotone"].to_numpy(bool)
    valid23 = recent["recent_games2_3_valid"].to_numpy(bool)
    valid45 = recent["recent_games4_5_valid"].to_numpy(bool)
    result = {
        "all": np.ones(len(frame), bool),
        "R": regular,
        "F": futures,
        "monotone": monotone,
        "R_monotone": regular & monotone,
        "F_monotone": futures & monotone,
        "nested_valid": valid23 & valid45,
        "R_nested_valid": regular & valid23 & valid45,
        "F_nested_valid": futures & valid23 & valid45,
    }
    threshold_specs = (
        ("n1_15", "recent1_reduced_n", 15.),
        ("n1_30", "recent1_reduced_n", 30.),
        ("n1_50", "recent1_reduced_n", 50.),
        ("n3_60", "recent3_reduced_n", 60.),
        ("n3_100", "recent3_reduced_n", 100.),
        ("n3_150", "recent3_reduced_n", 150.),
        ("n5_100", "recent5_reduced_n", 100.),
        ("n5_175", "recent5_reduced_n", 175.),
        ("n5_250", "recent5_reduced_n", 250.),
    )
    for name, column, threshold in threshold_specs:
        active = recent[column].to_numpy(float) >= threshold
        result[name] = active
        result[f"R_{name}"] = regular & active
        result[f"F_{name}"] = futures & active
        result[f"monotone_{name}"] = monotone & active
        result[f"R_monotone_{name}"] = regular & monotone & active
        result[f"F_monotone_{name}"] = futures & monotone & active
    return result


def main():
    usecols = [
        "season", "game_type", "asof_pitcher_success_rate",
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
    reference_members = {}
    for year in (2023, 2024):
        frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        recent = recent_window_features(frame)
        with np.load(
            ROOT / "research" / f"v58_recent_fraction_hl2_s1_{year}.npz"
        ) as archive:
            target = archive["target"].astype(float)
            base = np.clip(archive["base"].astype(float), .005, .995)
            extra = np.clip(archive["prediction"].astype(float), .005, .995)
            game_type = archive["game_type"].astype(str)
        paired_path = (
            ROOT / "research" / f"v35_lowcard_direct_hl2_s1_{year}.npz"
        )
        reference_path = paired_path if paired_path.exists() else (
            ROOT / "research" / f"v35_lowcard_direct_hl2_s3_{year}.npz"
        )
        reference_members[str(year)] = 1 if paired_path.exists() else 3
        with np.load(reference_path) as archive:
            reference = np.clip(archive["prediction"].astype(float), .005, .995)
        if not (
            len(frame) == len(target)
            and np.array_equal(frame["game_type"].astype(str), game_type)
        ):
            raise ValueError(f"row alignment failed for {year}")
        years[year] = {
            "target": target, "base": base, "game_type": game_type,
            # A paired model delta is safer here than blending the entire weak
            # direct model over v54.  Values are far from clipping boundaries.
            "direction": extra - reference,
            "cohorts": cohort_masks(frame, recent),
            "baseline": score(target, base, game_type),
        }

    names = sorted(set.intersection(*[
        set(values["cohorts"]) for values in years.values()
    ]))
    coefficients = {
        name: {
            year: gain_coefficients(
                values["target"], values["base"], values["direction"],
                values["game_type"], values["cohorts"][name],
            )
            for year, values in years.items()
        }
        for name in names
    }
    reports = []
    for name in names:
        for weight in np.round(np.arange(-2., 2.0001, .025), 4):
            gains = {}
            rows = {}
            for year, values in years.items():
                active = values["cohorts"][name]
                gains[str(year)] = {
                    key: linear * weight - quadratic * weight * weight
                    for key, (linear, quadratic) in coefficients[name][year].items()
                }
                rows[str(year)] = int(active.sum())
            year_gains = [gains[str(year)]["all"] for year in years]
            quarters = [
                gains[str(year)][f"q{quarter}"]
                for year in years for quarter in range(1, 5)
            ]
            halves = [
                gains[str(year)][half]
                for year in years for half in ("h1", "h2")
            ]
            regimes = [
                gains[str(year)][regime]
                for year in years for regime in ("R", "F")
            ]
            reports.append({
                "cohort": name, "weight": float(weight), "rows": rows,
                "gains": gains, "min_year": float(min(year_gains)),
                "mean_year": float(np.mean(year_gains)),
                "min_quarter": float(min(quarters)),
                "min_half": float(min(halves)),
                "min_regime": float(min(regimes)),
                "strict_floor": float(min(year_gains + quarters + halves + regimes)),
            })
    nonzero = [row for row in reports if abs(row["weight"]) > 1e-8]
    report = {
        "comparison": "v58 one-seed model minus otherwise identical v35 direct model",
        "v35_reference_members": reference_members,
        "baseline": {
            str(year): values["baseline"] for year, values in years.items()
        },
        "best_strict": sorted(
            nonzero,
            key=lambda row: (row["strict_floor"], row["mean_year"]),
            reverse=True,
        )[:100],
        "best_year_robust": sorted(
            nonzero,
            key=lambda row: (
                row["min_year"], row["min_quarter"], row["mean_year"],
            ), reverse=True,
        )[:100],
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research/v58_fraction_delta_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_strict": report["best_strict"][:15],
        "best_year_robust": report["best_year_robust"][:15],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
