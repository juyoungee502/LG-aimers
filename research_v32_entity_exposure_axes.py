"""Audit row-local pitcher/batter participation axes over the strong base.

Unlike entity residual tables, these directions contain no fitted target.  A
row only looks up how often its pitcher or batter appeared in the one or two
labelled seasons preceding the held-out season.  Centering and scaling are
also frozen from those labelled entities, never from evaluation rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
AMPLITUDES = np.round(np.arange(-.012, .01201, .00025), 6)


def masks(rows):
    position = np.arange(len(rows))
    result = {
        "all": np.ones(len(rows), dtype=bool),
        "R": rows["game_type"].eq("R").to_numpy(),
        "F": rows["game_type"].eq("F").to_numpy(),
    }
    for index, active in enumerate(np.array_split(position, 4), 1):
        mask = np.zeros(len(rows), dtype=bool)
        mask[active] = True
        result[f"q{index}"] = mask
    return result


def exposure_direction(raw, rows, year, entity, window, transform):
    history = raw.loc[
        raw["season"].between(year - window, year - 1), entity
    ]
    counts = history.value_counts(sort=False).astype(float)
    values = np.log1p(counts) if transform == "log" else counts
    centered = values - float(values.mean())
    scale = float(centered.std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError(f"degenerate {entity} exposure table for {year}")
    table = centered / scale
    # New entities carry no known participation effect.
    return rows[entity].map(table).fillna(0.).to_numpy(float)


def current_exposure_direction(history, rows, entity):
    column = "asof_pitcher_n" if entity == "pitcher_id" else "asof_batter_n"
    source = np.log1p(pd.to_numeric(history[column], errors="coerce").fillna(0.))
    mean = float(source.mean())
    std = float(source.std(ddof=0))
    values = np.log1p(pd.to_numeric(rows[column], errors="coerce").fillna(0.))
    return ((values - mean) / max(std, 1e-8)).to_numpy(float)


def gain_coefficients(target, base, direction, groups):
    """Exact convex-blend BSS gain: linear*a - quadratic*a**2."""
    residual = target - base
    result = {}
    for name, active in groups.items():
        rate = float(target[active].mean())
        denominator = rate * (1. - rate)
        result[name] = (
            200000. * float(np.mean(
                residual[active] * direction[active]
            )) / denominator,
            100000. * float(np.mean(
                direction[active] ** 2
            )) / denominator,
        )
    return result


def coefficient_gains(coefficients, amplitude):
    return {
        name: linear * amplitude - quadratic * amplitude ** 2
        for name, (linear, quadratic) in coefficients.items()
    }


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", low_memory=False,
        usecols=[
            "season", "game_type", "pitcher_id", "batter_id",
            "asof_pitcher_n", "asof_batter_n", "control_success",
        ],
    )
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as source:
        v23 = {key: source[key] for key in source.files}
    with np.load(
        ROOT / "research/rolling_2022_2024/v11_oof_predictions.npz"
    ) as source:
        v11 = {key: source[key] for key in source.files}

    years = {}
    for year in (2022, 2023, 2024):
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        archive = v11 if year == 2022 else v23
        active = archive["season"] == year
        target = archive["target"][active].astype(float)
        base = archive["blended"][active].astype(float)
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"archive alignment failed for {year}")
        direction = {}
        for entity_name, entity in (("pitcher", "pitcher_id"),
                                    ("batter", "batter_id")):
            for window in (1, 2, 3):
                for transform in ("linear", "log"):
                    name = f"{entity_name}_{transform}_w{window}"
                    direction[name] = exposure_direction(
                        raw, rows, year, entity, window, transform,
                    )
            direction[f"{entity_name}_current_log"] = (
                current_exposure_direction(
                    raw.loc[raw["season"].lt(year)], rows, entity,
                )
            )
        years[year] = {
            "rows": rows, "target": target, "base": base,
            "groups": masks(rows), "direction": direction,
        }

    gates = {
        "all": lambda values: values,
        "R": lambda values, rows=None: values * rows["game_type"].eq("R"),
        "F": lambda values, rows=None: values * rows["game_type"].eq("F"),
    }
    reports = {}
    names = list(years[2022]["direction"])
    for name in names:
        candidates = []
        for gate_name in gates:
            gated = {}
            for year, values in years.items():
                direction = values["direction"][name].copy()
                if gate_name == "R":
                    direction[~values["groups"]["R"]] = 0.
                elif gate_name == "F":
                    direction[~values["groups"]["F"]] = 0.
                gated[year] = direction
            coefficients = {
                year: gain_coefficients(
                    values["target"], values["base"], gated[year],
                    values["groups"],
                )
                for year, values in years.items()
            }
            for amplitude in AMPLITUDES:
                gains = {
                    str(year): coefficient_gains(
                        coefficients[year], amplitude,
                    )
                    for year, values in years.items()
                }
                core = [gains[str(year)]["all"] for year in years]
                quarters = [
                    gains[str(year)][f"q{index}"]
                    for year in years for index in range(1, 5)
                ]
                active_types = ("R", "F") if gate_name == "all" else (gate_name,)
                types = [
                    gains[str(year)][kind]
                    for year in years for kind in active_types
                ]
                candidates.append({
                    "gate": gate_name, "amplitude": float(amplitude),
                    "gains": gains, "min_year": min(core),
                    "mean_year": float(np.mean(core)),
                    "min_quarter": min(quarters), "min_type": min(types),
                    "strict_floor": min(core + quarters + types),
                })
        reports[name] = {
            "strict": sorted(
                candidates,
                key=lambda row: (row["strict_floor"], row["mean_year"]),
                reverse=True,
            )[:12],
            "year_robust": sorted(
                candidates,
                key=lambda row: (row["min_year"], row["mean_year"]),
                reverse=True,
            )[:12],
        }
        strict = reports[name]["strict"][0]
        robust = reports[name]["year_robust"][0]
        print(
            f"{name}: strict {strict['gate']} a={strict['amplitude']:+.5f} "
            f"floor={strict['strict_floor']:+.3f} "
            f"mean={strict['mean_year']:+.3f}; "
            f"year {robust['gate']} a={robust['amplitude']:+.5f} "
            f"min={robust['min_year']:+.3f} "
            f"mean={robust['mean_year']:+.3f} "
            f"q={robust['min_quarter']:+.3f}",
            flush=True,
        )

    output = ROOT / "research/v32_entity_exposure_axes.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
