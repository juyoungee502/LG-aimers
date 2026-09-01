"""Strict-forward audit of an original pitcher count-geometry correction.

For every validation year, the method constructs a 12-cell command profile for
each pitcher using only earlier official seasons.  Pitcher level and league
count difficulty are removed first; a low-rank decomposition then denoises the
remaining pitcher-by-count interaction.  Evaluation rows are only looked up by
their own pitcher and count state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"
CLIP = (0.005, 0.995)
DECAYS = (0.50, 0.75, 1.00)
SHRINKAGES = (100.0, 300.0, 1000.0, 3000.0)
RANKS = (1, 2, 3, 4, 6)
SCALES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(prediction, dtype=float), *CLIP)
    reference = float(y.mean() * (1.0 - y.mean()))
    return float(100_000.0 * (1.0 - np.mean(np.square(p - y)) / reference))


def gain(target: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(target, candidate) - bss(target, anchor))


def count_state(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(int) * 3
        + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(int)
    )


def build_geometry(
    history: pd.DataFrame,
    prediction_year: int,
    *,
    decay: float,
    shrinkage: float,
    rank: int,
) -> dict[str, object]:
    work = history[["season", "pitcher_id", "count_code", TARGET]].copy()
    age = (prediction_year - 1) - work["season"].to_numpy(int)
    work["weight"] = np.power(decay, np.maximum(age, 0))
    work["weighted_target"] = work["weight"] * work[TARGET]
    league_weight = float(work["weight"].sum())
    league_rate = float(work["weighted_target"].sum() / league_weight)

    pitcher = work.groupby("pitcher_id", sort=True, observed=True)[
        ["weight", "weighted_target"]
    ].sum()
    pitcher["rate"] = (
        pitcher["weighted_target"] + 500.0 * league_rate
    ) / (pitcher["weight"] + 500.0)
    count = work.groupby("count_code", sort=True, observed=True)[
        ["weight", "weighted_target"]
    ].sum().reindex(range(12), fill_value=0.0)
    count["rate"] = (
        count["weighted_target"] + 1000.0 * league_rate
    ) / (count["weight"] + 1000.0)

    cells = work.groupby(
        ["pitcher_id", "count_code"], sort=True, observed=True,
    )[["weight", "weighted_target"]].sum().reset_index()
    cells["pitcher_rate"] = cells["pitcher_id"].map(pitcher["rate"])
    cells["count_rate"] = cells["count_code"].map(count["rate"])
    additive = np.clip(
        cells["pitcher_rate"] + cells["count_rate"] - league_rate,
        0.05, 0.95,
    )
    observed = cells["weighted_target"] / cells["weight"]
    cells["interaction"] = (
        (observed - additive) * cells["weight"]
        / (cells["weight"] + shrinkage)
    )

    pitcher_ids = pitcher.index.to_numpy()
    pitcher_position = {value: index for index, value in enumerate(pitcher_ids)}
    matrix = np.zeros((len(pitcher_ids), 12), dtype=np.float64)
    exposure = np.zeros_like(matrix)
    row_index = cells["pitcher_id"].map(pitcher_position).to_numpy(int)
    column_index = cells["count_code"].to_numpy(int)
    matrix[row_index, column_index] = cells["interaction"].to_numpy(float)
    exposure[row_index, column_index] = cells["weight"].to_numpy(float)

    # The zeros are the empirical-Bayes prior for unobserved cells.  SVD shares
    # stable count-shape information between cells without introducing a player
    # or league-wide probability level.
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    use_rank = min(rank, len(singular))
    reconstructed = (
        u[:, :use_rank] * singular[:use_rank]
    ) @ vt[:use_rank]

    row_denominator = exposure.sum(axis=1)
    row_mean = np.divide(
        (reconstructed * exposure).sum(axis=1), row_denominator,
        out=np.zeros(len(reconstructed)), where=row_denominator > 0,
    )
    reconstructed -= row_mean[:, None]
    column_denominator = exposure.sum(axis=0)
    column_mean = np.divide(
        (reconstructed * exposure).sum(axis=0), column_denominator,
        out=np.zeros(12), where=column_denominator > 0,
    )
    reconstructed -= column_mean[None, :]
    global_mean = float(
        (reconstructed * exposure).sum() / max(exposure.sum(), 1.0)
    )
    reconstructed += global_mean
    return {
        "pitcher_ids": pitcher_ids,
        "values": reconstructed,
        "singular_values": singular,
        "league_rate": league_rate,
        "observed_cells": int(len(cells)),
    }


def apply_geometry(rows: pd.DataFrame, geometry: dict[str, object]) -> tuple[np.ndarray, float]:
    pitcher_ids = np.asarray(geometry["pitcher_ids"])
    values = np.asarray(geometry["values"], dtype=float)
    positions = pd.Index(pitcher_ids).get_indexer(rows["pitcher_id"])
    counts = rows["count_code"].to_numpy(int)
    known = positions >= 0
    correction = np.zeros(len(rows), dtype=np.float64)
    correction[known] = values[positions[known], counts[known]]
    return correction, float(known.mean())


def segment_report(
    target: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    regular: np.ndarray,
) -> dict[str, object]:
    positions = np.arange(len(target))
    return {
        "gain": gain(target, anchor, candidate),
        "half_gains": [
            gain(target[index], anchor[index], candidate[index])
            for index in np.array_split(positions, 2)
        ],
        "quarter_gains": [
            gain(target[index], anchor[index], candidate[index])
            for index in np.array_split(positions, 4)
        ],
        "regular_gain": gain(
            target[regular], anchor[regular], candidate[regular],
        ),
        "futures_gain": gain(
            target[~regular], anchor[~regular], candidate[~regular],
        ),
        "mean_absolute_change": float(np.mean(np.abs(candidate - anchor))),
    }


def pitcher_bootstrap(
    target: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    pitcher_id: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(target.mean() * (1.0 - target.mean()))
    improvement = np.square(anchor - target) - np.square(candidate - target)
    grouped = pd.DataFrame({
        "pitcher": pitcher_id.astype(str), "improvement": improvement,
    }).groupby("pitcher", sort=False)["improvement"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        width = min(64, repetitions - start)
        draw = rng.integers(0, len(grouped), size=(width, len(grouped)))
        samples[start:start + width] = (
            100_000.0 * sums[draw].sum(axis=1)
            / sizes[draw].sum(axis=1) / reference
        )
    return {
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "positive_probability": float(np.mean(samples > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=50000)
    args = parser.parse_args()
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "balls_before",
            "strikes_before", TARGET,
        ], encoding="utf-8-sig", low_memory=False,
    )
    raw["count_code"] = count_state(raw)
    with np.load(ROOT / "outputs/v65_oof_predictions.npz", allow_pickle=True) as archive:
        target = archive["target"].astype(float)
        season = archive["season"].astype(int)
        anchor = archive["blended"].astype(float)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(anchor) or not np.array_equal(
        rows[TARGET].to_numpy(float), target,
    ):
        raise ValueError("v65 OOF rows do not align with train.csv")

    directions: dict[tuple[float, float, int, int], np.ndarray] = {}
    geometry_audit: dict[str, object] = {}
    for year in (2023, 2024):
        history = raw.loc[raw["season"].lt(year)]
        validation = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for decay in DECAYS:
            for shrinkage in SHRINKAGES:
                for rank in RANKS:
                    geometry = build_geometry(
                        history, year, decay=decay,
                        shrinkage=shrinkage, rank=rank,
                    )
                    correction, coverage = apply_geometry(validation, geometry)
                    directions[(decay, shrinkage, rank, year)] = correction
                    geometry_audit[
                        f"d{decay}_k{int(shrinkage)}_r{rank}_{year}"
                    ] = {
                        "coverage": coverage,
                        "league_rate": float(geometry["league_rate"]),
                        "observed_cells": int(geometry["observed_cells"]),
                        "leading_singular_values": np.asarray(
                            geometry["singular_values"], dtype=float,
                        )[:6].tolist(),
                        "correction_std": float(correction.std()),
                    }

    candidates: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, int, float], np.ndarray] = {}
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    for decay in DECAYS:
        for shrinkage in SHRINKAGES:
            for rank in RANKS:
                direction = np.concatenate([
                    directions[(decay, shrinkage, rank, year)]
                    for year in (2023, 2024)
                ])
                for scale in SCALES:
                    prediction = np.clip(anchor + scale * direction, *CLIP)
                    reports = {
                        str(year): segment_report(
                            target[season == year], anchor[season == year],
                            prediction[season == year], regular[season == year],
                        ) for year in (2023, 2024)
                    }
                    all_halves = [
                        value for report in reports.values()
                        for value in report["half_gains"]
                    ]
                    all_quarters = [
                        value for report in reports.values()
                        for value in report["quarter_gains"]
                    ]
                    all_regimes = [
                        report[key] for report in reports.values()
                        for key in ("regular_gain", "futures_gain")
                    ]
                    stable = bool(
                        min(report["gain"] for report in reports.values()) > 0.0
                        and min(all_halves) > 0.0
                        and min(all_quarters) > 0.0
                        and min(all_regimes) > 0.0
                    )
                    key = (decay, shrinkage, rank, scale)
                    predictions[key] = prediction
                    candidates.append({
                        "decay": decay, "shrinkage": shrinkage,
                        "rank": rank, "scale": scale,
                        "reports": reports, "stable": stable,
                        "min_year_gain": float(min(
                            report["gain"] for report in reports.values()
                        )),
                        "mean_year_gain": float(np.mean([
                            report["gain"] for report in reports.values()
                        ])),
                        "min_half_gain": float(min(all_halves)),
                        "min_quarter_gain": float(min(all_quarters)),
                        "min_regime_gain": float(min(all_regimes)),
                    })
    candidates.sort(key=lambda item: (
        item["stable"], item["min_quarter_gain"], item["min_year_gain"],
        item["mean_year_gain"],
    ), reverse=True)
    # The absolute best worst-quarter value can be separated from a materially
    # stronger two-year result by less than one leaderboard point.  Treat those
    # values as statistically tied, then prefer the better worst-year result.
    stable_candidates = [item for item in candidates if item["stable"]]
    if stable_candidates:
        quarter_frontier = max(
            item["min_quarter_gain"] for item in stable_candidates
        )
        tied = [
            item for item in stable_candidates
            if item["min_quarter_gain"] >= quarter_frontier - 0.01
        ]
        best = max(tied, key=lambda item: (
            item["min_year_gain"], item["mean_year_gain"],
            item["min_half_gain"],
        ))
    else:
        best = candidates[0]
    best_key = (
        float(best["decay"]), float(best["shrinkage"]),
        int(best["rank"]), float(best["scale"]),
    )
    selected = predictions[best_key]
    bootstrap = {
        str(year): pitcher_bootstrap(
            target[season == year], anchor[season == year],
            selected[season == year],
            rows.loc[season == year, "pitcher_id"].to_numpy(),
            args.bootstrap, 675000 + year,
        ) for year in (2023, 2024)
    }
    strict_gate = bool(
        best["stable"]
        and all(value["positive_probability"] >= 0.85
                for value in bootstrap.values())
    )
    report = {
        "baseline": "v65_prediction_gap_meta_public_1135_0",
        "candidate": "original_low_rank_pitcher_count_geometry",
        "selected": best if strict_gate else None,
        "best": best,
        "bootstrap": bootstrap,
        "strict_gate": strict_gate,
        "top_candidates": candidates[:30],
        "geometry_audit": geometry_audit,
        "rules": {
            "strict_prior_seasons_only": True,
            "official_train_only": True,
            "external_reference_model_or_prediction_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v66_component_used": False,
        },
        "selection_policy": {
            "all_year_half_quarter_and_regime_gains_positive": True,
            "worst_quarter_tie_tolerance_bss": 0.01,
            "within_tie_prefer_worst_year_then_mean_year": True,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "research/v67_count_geometry.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/v67_count_geometry_oof.npz",
        target=target, season=season, anchor=anchor, blended=selected,
    )
    print(json.dumps({
        "candidate": report["candidate"], "best": best,
        "bootstrap": bootstrap, "strict_gate": strict_gate,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
