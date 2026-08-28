"""Screen leakage-safe season-environment corrections for cumulative rates.

For a validation season Y, every player history is built only from seasons
strictly before Y.  Historical successes are translated to the most recent
observed season's league environment, either globally or within game type.
The resulting player-rate correction is then shrunk by the row's current-
season exposure and added to the frozen v23 OOF prediction.

This is a research screen rather than a submission model.  It deliberately
audits both 2023 and 2024 forward folds before any feature is promoted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import training_history_arrays
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"


def segment_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    size = len(rows)
    position = np.arange(size)
    masks = {
        "all": np.ones(size, dtype=bool),
        "h1": position < size // 2,
        "h2": position >= size // 2,
        "R": rows["game_type"].eq("R").to_numpy(),
        "F": rows["game_type"].eq("F").to_numpy(),
    }
    for index, part in enumerate(np.array_split(position, 4), 1):
        active = np.zeros(size, dtype=bool)
        active[part] = True
        masks[f"q{index}"] = active
    return masks


def grouped_rates(
    seasonal: pd.DataFrame,
    id_col: str,
    year: int,
    adjustment: str,
    half_life: float | None,
    global_rates: dict[int, float],
    type_rates: dict[tuple[int, str], float],
) -> tuple[dict[int, float], dict[int, float]]:
    """Return raw and translated/recency-weighted history rates by entity."""
    history = seasonal.loc[seasonal["season"] < year].copy()
    reference_year = year - 1
    if half_life is None:
        history["weight"] = 1.0
    else:
        age = reference_year - history["season"].to_numpy(float)
        history["weight"] = np.power(.5, age / half_life)

    if adjustment == "none":
        offset = np.zeros(len(history), dtype=float)
    elif adjustment == "global":
        reference = global_rates[reference_year]
        offset = reference - history["season"].map(global_rates).to_numpy(float)
    elif adjustment == "game_type":
        reference = history["game_type"].map(
            lambda value: type_rates[(reference_year, str(value))]
        ).to_numpy(float)
        historical = np.fromiter(
            (
                type_rates[(int(season), str(game_type))]
                for season, game_type in zip(
                    history["season"], history["game_type"], strict=True
                )
            ),
            dtype=float,
            count=len(history),
        )
        offset = reference - historical
    else:
        raise ValueError(adjustment)

    history["weighted_n"] = history["n"] * history["weight"]
    history["weighted_s"] = (
        history["success"] + history["n"] * offset
    ) * history["weight"]
    adjusted = history.groupby(id_col, observed=True, sort=False)[
        ["weighted_n", "weighted_s"]
    ].sum()
    adjusted_rate = np.clip(
        adjusted["weighted_s"] / adjusted["weighted_n"], .005, .995
    )

    raw = history.groupby(id_col, observed=True, sort=False)[["n", "success"]].sum()
    raw_rate = raw["success"] / raw["n"]
    return raw_rate.to_dict(), adjusted_rate.to_dict()


def gain_geometry(
    target: np.ndarray,
    base: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    uncertainty = float(target[mask].mean() * (1. - target[mask].mean()))
    linear = 200000. * float(np.mean(
        (target[mask] - base[mask]) * direction[mask]
    )) / uncertainty
    quadratic = 100000. * float(np.mean(direction[mask] ** 2)) / uncertainty
    return linear, quadratic


def main() -> None:
    columns = [
        "season", "game_type", "pitcher_id", "batter_id", TARGET,
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_batter_n", "asof_batter_success_rate",
    ]
    raw = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw[TARGET].astype(np.float32)
    target_all = target_series.to_numpy(float)
    features = raw.drop(columns=[TARGET])
    p_base_n, p_base_s, b_base_n, b_base_s = training_history_arrays(
        features, target_series,
    )

    global_rates = raw.groupby("season", observed=True)[TARGET].mean().to_dict()
    type_rates = raw.groupby(
        ["season", "game_type"], observed=True,
    )[TARGET].mean().to_dict()
    seasonal_tables = {}
    for id_col in ("pitcher_id", "batter_id"):
        seasonal_tables[id_col] = raw.groupby(
            ["season", "game_type", id_col], observed=True, sort=False,
        )[TARGET].agg(n="size", success="sum").reset_index()

    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}

    fold_data: dict[int, dict[str, object]] = {}
    for year in (2023, 2024):
        raw_mask = raw["season"].eq(year).to_numpy()
        oof_mask = oof["season"] == year
        target = oof["target"][oof_mask].astype(float)
        if not np.allclose(target, target_all[raw_mask]):
            raise ValueError(f"v23 rows do not align for {year}")
        rows = raw.loc[raw_mask].reset_index(drop=True)
        fold_data[year] = {
            "raw_mask": raw_mask,
            "rows": rows,
            "target": target,
            "base": np.clip(oof["blended"][oof_mask].astype(float), .005, .995),
            "masks": segment_masks(rows),
        }

    reports = []
    diagnostics_by_config = {}
    half_lives: tuple[float | None, ...] = (None, 1., 2., 4.)
    mixtures = {
        "pitcher": (1., 0.),
        "p75_b25": (.75, .25),
        "p50_b50": (.50, .50),
        "p25_b75": (.25, .75),
        "batter": (0., 1.),
    }
    gates = ("all", "R", "F")
    strengths = (10., 25., 50., 100., 200.)

    for adjustment in ("none", "global", "game_type"):
        for half_life in half_lives:
            if adjustment == "none" and half_life is None:
                continue
            corrections: dict[int, dict[str, np.ndarray]] = {}
            diagnostics = {}
            for year, fold in fold_data.items():
                raw_mask = fold["raw_mask"]
                rows = fold["rows"]
                entity_corrections = {}
                diagnostics[str(year)] = {}
                for id_col in ("pitcher_id", "batter_id"):
                    old_map, adjusted_map = grouped_rates(
                        seasonal_tables[id_col], id_col, year, adjustment,
                        half_life, global_rates, type_rates,
                    )
                    old = rows[id_col].map(old_map).to_numpy(float)
                    adjusted = rows[id_col].map(adjusted_map).to_numpy(float)
                    correction = np.nan_to_num(adjusted - old, nan=0.)
                    entity_corrections[id_col] = correction

                    if id_col == "pitcher_id":
                        frozen_n = p_base_n[raw_mask].astype(float)
                        frozen_s = p_base_s[raw_mask].astype(float)
                    else:
                        frozen_n = b_base_n[raw_mask].astype(float)
                        frozen_s = b_base_s[raw_mask].astype(float)
                    frozen_rate = np.divide(
                        frozen_s, frozen_n,
                        out=np.full(len(rows), global_rates[year - 1], dtype=float),
                        where=frozen_n > 0,
                    )
                    historical = rows[id_col].map(old_map).to_numpy(float)
                    active = np.isfinite(historical) & (frozen_n > 0)
                    diagnostics[str(year)][id_col] = {
                        "coverage": float(active.mean()),
                        "raw_frozen_mae": float(np.mean(np.abs(
                            historical[active] - frozen_rate[active]
                        ))) if active.any() else None,
                        "mean_correction": float(np.mean(correction[active]))
                        if active.any() else 0.,
                    }
                corrections[year] = entity_corrections
            config_key = f"{adjustment}_hl{half_life}"
            diagnostics_by_config[config_key] = diagnostics

            for strength in strengths:
                directions = {}
                for year, fold in fold_data.items():
                    raw_mask = fold["raw_mask"]
                    rows = fold["rows"]
                    pitcher_correction = corrections[year]["pitcher_id"]
                    batter_correction = corrections[year]["batter_id"]

                    career_p_n = rows["asof_pitcher_n"].fillna(0).to_numpy(float)
                    career_b_n = rows["asof_batter_n"].fillna(0).to_numpy(float)
                    current_p_n = np.maximum(0., career_p_n - p_base_n[raw_mask])
                    current_b_n = np.maximum(0., career_b_n - b_base_n[raw_mask])
                    p_delta = strength / (current_p_n + strength) * pitcher_correction
                    b_delta = strength / (current_b_n + strength) * batter_correction
                    directions[year] = {name: p_weight * p_delta + b_weight * b_delta
                                        for name, (p_weight, b_weight) in mixtures.items()}

                for mixture in mixtures:
                    for gate in gates:
                        curves = {}
                        for year, fold in fold_data.items():
                            rows = fold["rows"]
                            direction = directions[year][mixture].copy()
                            if gate != "all":
                                direction *= rows["game_type"].eq(gate).to_numpy(float)
                            curves[str(year)] = {
                                name: gain_geometry(
                                    fold["target"], fold["base"], direction, mask,
                                )
                                for name, mask in fold["masks"].items() if mask.any()
                            }
                        for weight in np.round(np.arange(-2., 2.001, .05), 3):
                            if abs(weight) < 1e-9:
                                continue
                            gains = {
                                year: {
                                    name: linear * weight - quadratic * weight**2
                                    for name, (linear, quadratic) in year_curves.items()
                                }
                                for year, year_curves in curves.items()
                            }
                            time_segments = [
                                gains[year][name]
                                for year in ("2023", "2024")
                                for name in ("h1", "h2", "q1", "q2", "q3", "q4")
                            ]
                            reports.append({
                                "adjustment": adjustment,
                                "half_life": half_life,
                                "strength": strength,
                                "mixture": mixture,
                                "gate": gate,
                                "weight": float(weight),
                                "gains": gains,
                                "min_year": float(min(
                                    gains["2023"]["all"], gains["2024"]["all"]
                                )),
                                "mean_year": float(np.mean([
                                    gains["2023"]["all"], gains["2024"]["all"]
                                ])),
                                "min_time_segment": float(min(time_segments)),
                            })

    positive = [row for row in reports if row["min_year"] > 0.]
    rankings = {
        "best_maximin_year": sorted(
            positive,
            key=lambda row: (row["min_year"], row["min_time_segment"], row["mean_year"]),
            reverse=True,
        )[:50],
        "best_maximin_time": sorted(
            positive,
            key=lambda row: (row["min_time_segment"], row["min_year"], row["mean_year"]),
            reverse=True,
        )[:50],
        "best_mean": sorted(
            positive,
            key=lambda row: (row["mean_year"], row["min_year"], row["min_time_segment"]),
            reverse=True,
        )[:50],
    }
    baseline = {
        str(year): {
            name: float(bss(
                fold["target"][mask], fold["base"][mask]
            ))
            for name, mask in fold["masks"].items() if mask.any()
        }
        for year, fold in fold_data.items()
    }
    report = {
        "baseline_v23": baseline,
        "league_rates": {str(key): float(value) for key, value in global_rates.items()},
        "candidate_count": len(reports),
        "positive_both_years": len(positive),
        "diagnostics": diagnostics_by_config,
        "rankings": rankings,
    }
    output = ROOT / "research/v41_career_detrend.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    preview = {
        "baseline_v23": baseline,
        "league_rates": report["league_rates"],
        "candidate_count": len(reports),
        "positive_both_years": len(positive),
        "rankings": {key: value[:8] for key, value in rankings.items()},
    }
    print(json.dumps(preview, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
