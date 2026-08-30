"""Independent hierarchical residual model audited as a v64 complement.

This is an original, clean-room implementation of the public high-score idea:
predict the residual around a reliability-shrunk pitcher baseline with several
recency-weighted CatBoost regressors.  It uses only the official training data
and the existing row-local feature builder.  The externally published OOF and
weights are not used for fitting this model.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
CLIP = (0.005, 0.995)
CATEGORICAL = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
]
SPECS = (
    {"name": "recent_d6", "decay": 0.55, "depth": 6, "iterations": 420, "seed": 65061},
    {"name": "recent_d8", "decay": 0.55, "depth": 8, "iterations": 260, "seed": 65082},
    {"name": "latest_d7", "decay": 0.30, "depth": 7, "iterations": 320, "seed": 65073},
)
FIXED_SCALES = {"R": 0.30, "F": 0.10}
NEIGHBORHOOD = (
    (0.25, 0.075), (0.25, 0.10), (0.30, 0.075),
    (0.30, 0.10), (0.35, 0.10), (0.30, 0.125),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def hierarchical_base(raw: pd.DataFrame, features: pd.DataFrame, prior: float) -> np.ndarray:
    recent = raw[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].apply(pd.to_numeric, errors="coerce")
    recent_std = recent.std(axis=1).fillna(0.15).clip(0.0, 0.5).to_numpy(float)
    career_n = pd.to_numeric(raw["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    career_rate = pd.to_numeric(
        raw["asof_pitcher_success_rate"], errors="coerce",
    ).fillna(prior).to_numpy(float)
    dynamic_strength = np.clip(
        55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(career_n)),
        50.0, 180.0,
    )
    career = (
        career_rate * career_n + prior * dynamic_strength
    ) / (career_n + dynamic_strength)
    season_n = features["pitcher_season_n"].to_numpy(float)
    season_raw = features["pitcher_season_success_rate"].to_numpy(float)
    season_raw = np.where(season_n > 0.0, season_raw, prior)
    season_estimate = (season_raw * season_n + 30.0 * prior) / (season_n + 30.0)
    reliability = season_n / (season_n + 80.0)
    return career + (0.15 + 0.30 * reliability) * (season_estimate - career)


def model_parameters(spec: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {
        "iterations": int(spec["iterations"]),
        "depth": int(spec["depth"]),
        "learning_rate": 0.025,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 60.0,
        "random_strength": 0.35,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "random_seed": int(spec["seed"]),
        "task_type": args.task_type,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if args.task_type == "GPU":
        result.update(devices=args.devices, border_count=32, gpu_ram_part=0.85)
    return result


def weighted_prior(target: np.ndarray, seasons: np.ndarray, target_year: int, decay: float) -> float:
    weights = np.power(decay, (target_year - 1) - seasons)
    return float(np.average(target, weights=weights))


def fit_fold(
    features: pd.DataFrame,
    raw: pd.DataFrame,
    target: np.ndarray,
    seasons: np.ndarray,
    target_year: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    train_mask = seasons < target_year
    valid_mask = seasons == target_year
    members: list[np.ndarray] = []
    audit: dict[str, object] = {}
    for spec in SPECS:
        decay = float(spec["decay"])
        prior = weighted_prior(
            target[train_mask], seasons[train_mask], target_year, decay,
        )
        base = hierarchical_base(raw, features, prior)
        residual = target[train_mask] - base[train_mask]
        weights = np.power(
            decay, (target_year - 1) - seasons[train_mask],
        )
        model = CatBoostRegressor(**model_parameters(spec, args))
        model.fit(
            features.loc[train_mask], residual,
            sample_weight=weights, cat_features=CATEGORICAL,
        )
        prediction = np.clip(
            base[valid_mask] + model.predict(features.loc[valid_mask]), *CLIP,
        )
        members.append(prediction)
        audit[str(spec["name"])] = {
            "prior": prior,
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "score": float(bss(target[valid_mask], prediction)),
        }
    return np.mean(members, axis=0), audit


def forward_affine_calibration(
    source_prediction: np.ndarray,
    source_target: np.ndarray,
    source_season: np.ndarray,
    target_year: int,
    prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.power(0.55, (target_year - 1) - source_season)
    mean_prediction = float(np.average(source_prediction, weights=weights))
    mean_target = float(np.average(source_target, weights=weights))
    centered = source_prediction - mean_prediction
    denominator = float(np.sum(weights * np.square(centered)))
    slope = float(
        np.sum(weights * centered * (source_target - mean_target))
        / max(denominator, 1e-12)
    )
    slope = float(np.clip(slope, 0.25, 1.25))
    calibrated = np.clip(
        mean_target + slope * (prediction - mean_prediction), *CLIP,
    )
    return calibrated, {
        "source_prediction_mean": mean_prediction,
        "source_target_mean": mean_target,
        "slope": slope,
        "source_year_min": int(source_season.min()),
        "source_year_max": int(source_season.max()),
    }


def gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(y, candidate) - bss(y, base))


def candidate_prediction(
    base: np.ndarray,
    alternative: np.ndarray,
    regular: np.ndarray,
    r_scale: float,
    f_scale: float,
) -> np.ndarray:
    scale = np.where(regular, r_scale, f_scale)
    return np.clip(base + scale * (alternative - base), *CLIP)


def segment_report(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    regular: np.ndarray,
) -> dict[str, object]:
    positions = np.arange(len(y))
    halves = np.array_split(positions, 2)
    quarters = np.array_split(positions, 4)
    return {
        "gain": gain(y, base, candidate),
        "half_gains": [gain(y[i], base[i], candidate[i]) for i in halves],
        "quarter_gains": [gain(y[i], base[i], candidate[i]) for i in quarters],
        "group_gains": {
            "R": gain(y[regular], base[regular], candidate[regular]),
            "F": gain(y[~regular], base[~regular], candidate[~regular]),
        },
        "mean_absolute_change": float(np.mean(np.abs(candidate - base))),
    }


def cluster_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(y.mean() * (1.0 - y.mean()))
    row_gain = np.square(base - y) - np.square(candidate - y)
    grouped = pd.DataFrame({
        "pitcher": pitcher.astype(str), "gain": row_gain,
    }).groupby("pitcher", sort=False)["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        sampled = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start:start + count] = (
            100_000.0 * sums[sampled].sum(axis=1)
            / sizes[sampled].sum(axis=1) / reference
        )
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def main() -> None:
    args = arguments()
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    seasons_all = raw["season"].to_numpy(int)
    history = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *history, global_prior=float(target_series.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in CATEGORICAL:
        features[column] = features[column].fillna(-1).astype(str)

    active = np.isin(seasons_all, [2023, 2024])
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        y = archive["target"].astype(float)
        v64 = archive["blended"].astype(float)
        seasons = archive["season"].astype(int)
    if not np.array_equal(y, target_all[active]):
        raise ValueError("v64 OOF and official rows are not aligned")

    alternatives: dict[int, np.ndarray] = {}
    folds: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        prediction, audit = fit_fold(
            features, raw, target_all, seasons_all, year, args,
        )
        alternatives[year] = prediction
        folds[str(year)] = audit

    calibrated: dict[int, np.ndarray] = {}
    calibration: dict[str, object] = {}
    for year in (2023, 2024):
        source_years = [candidate for candidate in alternatives if candidate < year]
        source_prediction = np.concatenate([alternatives[candidate] for candidate in source_years])
        source_target = np.concatenate([
            target_all[seasons_all == candidate] for candidate in source_years
        ])
        source_season = np.concatenate([
            np.full(len(alternatives[candidate]), candidate, dtype=int)
            for candidate in source_years
        ])
        calibrated[year], calibration[str(year)] = forward_affine_calibration(
            source_prediction, source_target, source_season, year,
            alternatives[year],
        )
    alternative = np.concatenate([calibrated[2023], calibrated[2024]])
    active_rows = raw.loc[active].reset_index(drop=True)
    regular = active_rows["game_type"].astype(str).eq("R").to_numpy()

    sensitivity: list[dict[str, object]] = []
    for r_scale, f_scale in NEIGHBORHOOD:
        prediction = candidate_prediction(
            v64, alternative, regular, r_scale, f_scale,
        )
        reports = {}
        for year in (2023, 2024):
            mask = seasons == year
            reports[str(year)] = segment_report(
                y[mask], v64[mask], prediction[mask], regular[mask],
            )
        sensitivity.append({
            "r_scale": r_scale, "f_scale": f_scale, "years": reports,
            "minimum_year_gain": float(min(
                reports["2023"]["gain"], reports["2024"]["gain"],
            )),
            "minimum_segment_gain": float(min(
                gain_value
                for report in reports.values()
                for gain_value in (
                    report["half_gains"] + list(report["group_gains"].values())
                )
            )),
        })

    selected_prediction = candidate_prediction(
        v64, alternative, regular, FIXED_SCALES["R"], FIXED_SCALES["F"],
    )
    fixed = next(
        row for row in sensitivity
        if row["r_scale"] == FIXED_SCALES["R"]
        and row["f_scale"] == FIXED_SCALES["F"]
    )
    bootstrap = {
        str(year): cluster_bootstrap(
            y[seasons == year], v64[seasons == year],
            selected_prediction[seasons == year],
            active_rows.loc[seasons == year, "pitcher_id"].to_numpy(),
            args.bootstrap, 655000 + year,
        ) for year in (2023, 2024)
    }
    strict_gate = bool(
        fixed["minimum_year_gain"] > 0.0
        and fixed["minimum_segment_gain"] >= 0.0
        and min(row["minimum_year_gain"] for row in sensitivity) > 0.0
        and min(item["ci_low"] for item in bootstrap.values()) > 0.0
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "candidate": "clean_room_hierarchical_residual_ensemble",
        "fixed_scales_from_public_oof_audit": FIXED_SCALES,
        "alternative_scores": {
            str(year): float(bss(y[seasons == year], alternative[seasons == year]))
            for year in (2023, 2024)
        },
        "fold_models": folds,
        "forward_affine_calibration": calibration,
        "fixed_candidate": fixed,
        "scale_neighborhood": sensitivity,
        "bootstrap": bootstrap,
        "strict_gate": strict_gate,
        "rules": {
            "official_data_only_for_fit": True,
            "external_prediction_or_model_used_for_fit": False,
            "row_independent_inference": True,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "research/v65_hierarchical_independent.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/v65_hierarchical_independent_oof.npz",
        target=y, season=seasons, alternative=alternative,
        blended=selected_prediction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
