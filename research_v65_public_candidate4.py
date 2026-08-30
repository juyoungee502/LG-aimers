"""Audit a public ExtraTrees/Beta-Binomial family over the v64 OOF.

The structure was published by a member of a current 1150+ leaderboard team.
No published prediction, fitted model, test statistic, or external data is used:
all candidates are rebuilt from the official train.csv with strict forward
folds.  This file is research-only; production training is promoted separately
after the temporal and cluster-bootstrap gates pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"
ID_COLUMN = "row_id"
CATEGORICAL = ["top_bottom", "game_type", "base_state"]
EPSILON = 1e-6
EXTRA_WEIGHT = 0.7590660319
WEIGHT_GRID = tuple(np.round(np.arange(0.0, 0.2001, 0.0125), 4))
BETA_NAMES = [
    "pitcher_season_posterior",
    "batter_season_posterior",
    "previous_season_game_type_prior",
    "pitcher_career_rate",
    "pitcher_prev5_game_rate",
]


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    reference = float(target.mean() * (1.0 - target.mean()))
    return float(100_000.0 * (1.0 - np.mean(np.square(prediction - target)) / reference))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def add_time_safe_features(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild current-season states using only official as-of rows and prior seasons."""
    enriched = data.copy()
    for entity, id_column, n_column, rate_column in (
        ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ):
        totals = (
            data.groupby([id_column, "season"], observed=True)[TARGET]
            .agg(season_n="size", season_successes="sum")
            .reset_index()
            .sort_values([id_column, "season"])
        )
        totals["history_n"] = totals.groupby(id_column, observed=True)["season_n"].cumsum() - totals["season_n"]
        totals["history_successes"] = (
            totals.groupby(id_column, observed=True)["season_successes"].cumsum()
            - totals["season_successes"]
        )
        lookup = totals.set_index([id_column, "season"])[["history_n", "history_successes"]]
        keys = pd.MultiIndex.from_frame(enriched[[id_column, "season"]])
        history = lookup.reindex(keys)
        history_n = history["history_n"].fillna(0.0).to_numpy(dtype=np.float64)
        history_successes = history["history_successes"].fillna(0.0).to_numpy(dtype=np.float64)
        asof_n = enriched[n_column].to_numpy(dtype=np.float64)
        asof_successes = enriched[rate_column].fillna(0.0).to_numpy(dtype=np.float64) * asof_n
        season_n = np.maximum(asof_n - history_n, 0.0)
        season_successes = np.clip(asof_successes - history_successes, 0.0, season_n)
        history_rate = np.divide(
            history_successes,
            history_n,
            out=np.full_like(history_successes, np.nan),
            where=history_n > 0,
        )
        season_rate = np.divide(
            season_successes,
            season_n,
            out=np.full_like(season_successes, np.nan),
            where=season_n > 0,
        )
        enriched[f"derived_{entity}_history_n"] = history_n
        enriched[f"derived_{entity}_history_success_rate"] = history_rate
        enriched[f"derived_{entity}_season_n"] = season_n
        enriched[f"derived_{entity}_season_success_rate"] = season_rate
        for strength in (10.0, 50.0, 200.0):
            fallback = np.where(np.isfinite(history_rate), history_rate, 0.5)
            enriched[f"derived_{entity}_season_success_rate_s{int(strength)}"] = (
                season_successes + strength * fallback
            ) / (season_n + strength)

    enriched["derived_prev1_minus_career_success"] = (
        enriched["asof_pitcher_prev1_game_success_rate"] - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev3_minus_career_success"] = (
        enriched["asof_pitcher_prev3_game_success_rate"] - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev5_minus_career_success"] = (
        enriched["asof_pitcher_prev5_game_success_rate"] - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_recent_success_slope"] = (
        enriched["asof_pitcher_prev1_game_success_rate"]
        - enriched["asof_pitcher_prev5_game_success_rate"]
    )
    enriched["derived_two_strike"] = enriched["strikes_before"].eq(2).astype("int8")
    enriched["derived_three_ball"] = enriched["balls_before"].eq(3).astype("int8")
    enriched["derived_full_count"] = (
        enriched["balls_before"].eq(3) & enriched["strikes_before"].eq(2)
    ).astype("int8")
    enriched["derived_platoon_same_hand"] = (
        enriched["pitcher_hand"] == enriched["batter_hand"]
    ).astype("int8")
    enriched["derived_futures_new_regime"] = (
        enriched["game_type"].eq("F") & enriched["season"].ge(2023)
    ).astype("int8")
    return enriched


def make_extra_pipeline(feature_columns: list[str], estimators: int, workers: int, seed: int) -> Pipeline:
    numeric = [column for column in feature_columns if column not in CATEGORICAL]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        verbose_feature_names_out=False,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=estimators,
        max_depth=14,
        min_samples_leaf=100,
        max_features=0.8,
        n_jobs=workers,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def beta_candidates(data: pd.DataFrame, season: int, concentration: float) -> tuple[np.ndarray, np.ndarray]:
    mask = data["season"].eq(season).to_numpy()
    frame = data.loc[mask]
    previous = data.loc[data["season"].eq(season - 1)]
    prior_table = previous.groupby("game_type", observed=True)[TARGET].mean()
    global_prior = float(data.loc[data["season"].lt(season), TARGET].mean())
    prior = frame["game_type"].map(prior_table).fillna(global_prior).to_numpy(dtype=np.float64)
    if season == 2023:
        prior = np.where(frame["game_type"].eq("F").to_numpy(), 0.5, prior)

    posteriors: list[np.ndarray] = []
    for entity in ("pitcher", "batter"):
        n = frame[f"derived_{entity}_season_n"].to_numpy(dtype=np.float64)
        rate = frame[f"derived_{entity}_season_success_rate"].to_numpy(dtype=np.float64)
        successes = np.rint(np.nan_to_num(rate, nan=0.0) * n)
        posteriors.append((successes + concentration * prior) / (n + concentration))
    career = frame["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    career = np.where(np.isfinite(career), career, prior)
    recent = frame["asof_pitcher_prev5_game_success_rate"].to_numpy(dtype=np.float64)
    recent = np.where(np.isfinite(recent), recent, career)
    return mask, np.column_stack([posteriors[0], posteriors[1], prior, career, recent])


def tune_beta(data: pd.DataFrame, prediction_season: int) -> tuple[dict[str, object], np.ndarray]:
    calibration_season = prediction_season - 1
    best: tuple[float, float, np.ndarray] | None = None
    for concentration in (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0):
        calibration_mask, candidates = beta_candidates(data, calibration_season, concentration)
        target = data.loc[calibration_mask, TARGET].to_numpy(dtype=np.float64)
        result = minimize(
            lambda weights: float(np.mean(np.square(candidates @ weights - target))),
            x0=np.full(candidates.shape[1], 1.0 / candidates.shape[1]),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * candidates.shape[1],
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(result.message)
        candidate = (float(result.fun), concentration, np.asarray(result.x, dtype=np.float64))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No beta-binomial fit")
    calibration_brier, concentration, weights = best
    _, candidates = beta_candidates(data, prediction_season, concentration)
    artifact = {
        "calibration_season": calibration_season,
        "concentration": concentration,
        "candidate_names": BETA_NAMES,
        "weights": weights.tolist(),
        "calibration_brier": calibration_brier,
    }
    return artifact, np.clip(candidates @ weights, EPSILON, 1.0 - EPSILON)


def fit_logit_calibration(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    transformed = logit(probability)
    result = minimize(
        lambda parameters: float(
            np.mean(np.square(sigmoid(parameters[0] * transformed + parameters[1]) - target))
        ),
        x0=np.asarray([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((0.25, 2.5), (-1.0, 1.0)),
        options={"maxiter": 300, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.x[1])


def calibrate_by_game_type(current: dict[str, np.ndarray], previous: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, list[float]]]:
    output = np.asarray(current["raw"], dtype=np.float64).copy()
    settings: dict[str, list[float]] = {}
    for group in ("R", "F"):
        source = np.asarray(previous["game_type"]).astype(str) == group
        destination = np.asarray(current["game_type"]).astype(str) == group
        slope, intercept = fit_logit_calibration(
            np.asarray(previous["target"], dtype=np.float64)[source],
            np.asarray(previous["raw"], dtype=np.float64)[source],
        )
        output[destination] = sigmoid(slope * logit(output[destination]) + intercept)
        settings[group] = [slope, intercept]
    return np.clip(output, EPSILON, 1.0 - EPSILON), settings


def gain(target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> float:
    return bss(target, candidate) - bss(target, baseline)


def select_weight(
    folds: dict[int, dict[str, np.ndarray]],
    candidate_name: str,
    group: str,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for weight in WEIGHT_GRID:
        gains: list[float] = []
        for year, fold in sorted(folds.items()):
            mask = np.asarray(fold["game_type"]).astype(str) == group
            base = np.asarray(fold["baseline"])[mask]
            other = np.asarray(fold[candidate_name])[mask]
            prediction = (1.0 - weight) * base + weight * other
            gains.append(gain(np.asarray(fold["target"])[mask], base, prediction))
        objective = float(np.mean(gains) + 0.75 * np.min(gains))
        rows.append(
            {
                "candidate": candidate_name,
                "group": group,
                "weight": float(weight),
                "objective": objective,
                "min_year_gain": float(np.min(gains)),
                "mean_year_gain": float(np.mean(gains)),
                **{f"gain_{year}": value for year, value in zip(sorted(folds), gains)},
            }
        )
    rows.sort(key=lambda row: (float(row["objective"]), -float(row["weight"])), reverse=True)
    best = rows[0]
    if float(best["min_year_gain"]) <= 0.0:
        return 0.0, rows
    return float(best["weight"]), rows


def combined_prediction(fold: dict[str, np.ndarray], candidate_name: str, r_weight: float, f_weight: float) -> np.ndarray:
    game_type = np.asarray(fold["game_type"]).astype(str)
    weights = np.where(game_type == "F", f_weight, r_weight)
    return np.clip(
        (1.0 - weights) * np.asarray(fold["baseline"], dtype=np.float64)
        + weights * np.asarray(fold[candidate_name], dtype=np.float64),
        EPSILON,
        1.0 - EPSILON,
    )


def cluster_bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    reference = float(target.mean() * (1.0 - target.mean()))
    row_gain = np.square(baseline - target) - np.square(candidate - target)
    frame = pd.DataFrame({"cluster": np.asarray(clusters).astype(str), "gain": row_gain})
    aggregate = frame.groupby("cluster", sort=False)["gain"].agg(["sum", "size"])
    sums = aggregate["sum"].to_numpy(dtype=np.float64)
    sizes = aggregate["size"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        indices = rng.integers(0, len(aggregate), size=(count, len(aggregate)))
        samples[start : start + count] = (
            100_000.0 * sums[indices].sum(axis=1) / sizes[indices].sum(axis=1) / reference
        )
    return {
        "delta_bss": float(100_000.0 * row_gain.mean() / reference),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "positive_probability": float(np.mean(samples > 0.0)),
        "clusters": int(len(aggregate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    print("loading official train.csv", flush=True)
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    data = add_time_safe_features(raw)
    feature_columns = [column for column in data.columns if column not in {ID_COLUMN, TARGET}]
    folds: dict[int, dict[str, np.ndarray]] = {}
    models: dict[int, dict[str, np.ndarray]] = {}
    beta_artifacts: dict[str, object] = {}

    for year in (2022, 2023, 2024):
        train_mask = data["season"].lt(year).to_numpy()
        valid_mask = data["season"].eq(year).to_numpy()
        pipeline = make_extra_pipeline(feature_columns, args.estimators, args.workers, args.seed)
        print(f"fit ExtraTrees {year}: train={train_mask.sum():,} valid={valid_mask.sum():,}", flush=True)
        pipeline.fit(
            data.loc[train_mask, feature_columns],
            data.loc[train_mask, TARGET].to_numpy(dtype=np.uint8),
        )
        extra = pipeline.predict_proba(data.loc[valid_mask, feature_columns])[:, 1]
        beta_artifact, beta = tune_beta(data, year)
        raw_candidate = np.clip(EXTRA_WEIGHT * extra + (1.0 - EXTRA_WEIGHT) * beta, EPSILON, 1.0 - EPSILON)
        models[year] = {
            "target": data.loc[valid_mask, TARGET].to_numpy(dtype=np.float64),
            "game_type": data.loc[valid_mask, "game_type"].astype(str).to_numpy(),
            "pitcher_id": data.loc[valid_mask, "pitcher_id"].astype(str).to_numpy(),
            "extra": extra,
            "beta": beta,
            "raw": raw_candidate,
        }
        beta_artifacts[str(year)] = beta_artifact
        del pipeline

    calibrations: dict[str, object] = {}
    for year in (2023, 2024):
        calibrated, setting = calibrate_by_game_type(models[year], models[year - 1])
        models[year]["calibrated"] = calibrated
        calibrations[str(year)] = {"source_year": year - 1, "by_game_type": setting}

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        seasons = archive["season"].astype(int)
        v64_target = archive["target"].astype(np.float64)
        v64_prediction = archive["blended"].astype(np.float64)
    expected_target = data.loc[data["season"].isin([2023, 2024]), TARGET].to_numpy(dtype=np.float64)
    if not np.array_equal(v64_target, expected_target):
        raise ValueError("v64 OOF is not aligned with official training rows")

    for year in (2023, 2024):
        mask = seasons == year
        model = models[year]
        if not np.array_equal(v64_target[mask], model["target"]):
            raise ValueError(f"v64/Candidate4 target mismatch in {year}")
        folds[year] = {**model, "baseline": v64_prediction[mask]}

    family_summaries: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for candidate_name in ("extra", "beta", "raw", "calibrated"):
        r_weight, r_rows = select_weight(folds, candidate_name, "R")
        f_weight, f_rows = select_weight(folds, candidate_name, "F")
        weight_rows.extend(r_rows + f_rows)
        year_gains: dict[str, float] = {}
        half_gains: dict[str, list[float]] = {}
        quarter_gains: dict[str, list[float]] = {}
        candidate_predictions: dict[int, np.ndarray] = {}
        for year, fold in sorted(folds.items()):
            prediction = combined_prediction(fold, candidate_name, r_weight, f_weight)
            candidate_predictions[year] = prediction
            target = np.asarray(fold["target"])
            baseline = np.asarray(fold["baseline"])
            year_gains[str(year)] = gain(target, baseline, prediction)
            halves = np.array_split(np.arange(len(target)), 2)
            quarters = np.array_split(np.arange(len(target)), 4)
            half_gains[str(year)] = [gain(target[index], baseline[index], prediction[index]) for index in halves]
            quarter_gains[str(year)] = [gain(target[index], baseline[index], prediction[index]) for index in quarters]
        latest = folds[2024]
        bootstrap = cluster_bootstrap(
            latest["target"],
            latest["baseline"],
            candidate_predictions[2024],
            latest["pitcher_id"],
            args.bootstrap,
            args.seed + 2024,
        )
        family_summaries.append(
            {
                "candidate": candidate_name,
                "r_weight": r_weight,
                "f_weight": f_weight,
                "year_gains": year_gains,
                "half_gains": half_gains,
                "quarter_gains": quarter_gains,
                "bootstrap_2024": bootstrap,
                "mean_absolute_change_2024": float(
                    np.mean(np.abs(candidate_predictions[2024] - latest["baseline"]))
                ),
                "prediction_correlation_2024": float(
                    np.corrcoef(candidate_predictions[2024], latest["baseline"])[0, 1]
                ),
                "strict_gate": bool(
                    min(year_gains.values()) > 0.0
                    and min(value for values in half_gains.values() for value in values) >= 0.0
                    and bootstrap["ci_low"] > 0.0
                ),
            }
        )

    family_summaries.sort(
        key=lambda row: (
            bool(row["strict_gate"]),
            min(row["year_gains"].values()),
            np.mean(list(row["year_gains"].values())),
        ),
        reverse=True,
    )
    selected = family_summaries[0]
    report = {
        "source": {
            "repository": "Jungminii-1114/LG_AIMERS_personal_repo",
            "method": "Candidate4 ExtraTrees + Beta-Binomial",
            "external_model_or_prediction_used": False,
        },
        "baseline": "v64_public_method_transfer",
        "configuration": {
            "estimators": args.estimators,
            "max_depth": 14,
            "min_samples_leaf": 100,
            "extra_weight": EXTRA_WEIGHT,
        },
        "families": family_summaries,
        "selected": selected if selected["strict_gate"] else None,
        "calibrations": calibrations,
        "beta_artifacts": beta_artifacts,
        "runtime_seconds": time.perf_counter() - started,
        "rules": {
            "official_data_only": True,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    research = ROOT / "research"
    research.mkdir(exist_ok=True)
    (research / "v65_public_candidate4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(weight_rows).to_csv(research / "v65_public_candidate4_weights.csv", index=False)
    np.savez_compressed(
        ROOT / "outputs/v65_public_candidate4_oof.npz",
        target=np.concatenate([folds[year]["target"] for year in (2023, 2024)]),
        season=np.concatenate([
            np.full(len(folds[year]["target"]), year, dtype=np.int16) for year in (2023, 2024)
        ]),
        game_type=np.concatenate([folds[year]["game_type"] for year in (2023, 2024)]),
        pitcher_id=np.concatenate([folds[year]["pitcher_id"] for year in (2023, 2024)]),
        baseline=np.concatenate([folds[year]["baseline"] for year in (2023, 2024)]),
        extra=np.concatenate([folds[year]["extra"] for year in (2023, 2024)]),
        beta=np.concatenate([folds[year]["beta"] for year in (2023, 2024)]),
        raw=np.concatenate([folds[year]["raw"] for year in (2023, 2024)]),
        calibrated=np.concatenate([folds[year]["calibrated"] for year in (2023, 2024)]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
