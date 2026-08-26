"""Train a rolling-validated ensemble optimized for the official Brier score."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize

from feature_engineering import (
    ID_COL, TARGET_COL, add_state_interactions, add_training_component_features,
    build_end_history, engineer_features, training_history_arrays,
)

CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
MODEL_NAMES = [
    "lgb_a", "lgb_b", "catboost", "history_expert", "count_expert",
    "categorical_catboost", "categorical_count_expert", "brier_regressor",
]
SEGMENT_MODEL_INDICES = [2, 4, 5, 6, 7]
SEGMENT_MODEL_NAMES = [MODEL_NAMES[index] for index in SEGMENT_MODEL_INDICES]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="submit/model")
    p.add_argument("--diagnostic-dir", default="outputs")
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--preset", choices=("fast", "full"), default="full")
    p.add_argument("--seed", type=int, default=20260826)
    return p.parse_args()


def season_weights(seasons: pd.Series, reference: int) -> np.ndarray:
    age = np.maximum(0, reference - seasons.to_numpy(np.float32))
    return np.exp(-math.log(2) * age / 3.0).astype(np.float32)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray) -> float:
    r = float(np.mean(y)); reference = r * (1 - r)
    return max(0.0, 100000.0 * (1.0 - brier(y, p) / reference))


def history_expert(x: pd.DataFrame, prior: float) -> np.ndarray:
    """Stable row-only probability expert based on official as-of rates."""
    specs = [
        ("asof_pitcher_prev1_game_success_rate", .12),
        ("asof_pitcher_prev3_game_success_rate", .28),
        ("asof_pitcher_prev5_game_success_rate", .20),
        ("pitcher_season_success_s100", .22),
        ("asof_pitcher_success_rate", .10),
        ("asof_batter_success_rate", .08),
    ]
    total = np.zeros(len(x), np.float64); weight = np.zeros(len(x), np.float64)
    for col, w in specs:
        values = x[col].to_numpy(np.float64)
        ok = np.isfinite(values)
        total[ok] += w * values[ok]; weight[ok] += w
    return np.divide(total, weight, out=np.full(len(x), prior), where=weight > 0)


def lgb_params(variant: int, seed: int, threads: int) -> dict:
    common = dict(
        objective="binary", metric="None", learning_rate=.025,
        verbosity=-1, force_col_wise=True, seed=seed, num_threads=threads,
        max_bin=127, feature_pre_filter=False,
    )
    if variant == 0:
        common.update(num_leaves=31, min_data_in_leaf=900, feature_fraction=.84,
                      bagging_fraction=.86, bagging_freq=1, lambda_l1=1., lambda_l2=14.)
    else:
        common.update(num_leaves=47, min_data_in_leaf=1400, feature_fraction=.72,
                      bagging_fraction=.80, bagging_freq=1, lambda_l1=2., lambda_l2=20.)
    return common


def lgb_brier(pred: np.ndarray, dataset: lgb.Dataset):
    return "brier", brier(dataset.get_label(), pred), False


def fit_lgb(x_train, y_train, x_valid, y_valid, weights, variant, args):
    rounds = 700 if args.preset == "full" else 100
    train_set = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)
    return lgb.train(
        lgb_params(variant, args.seed + variant * 97, args.threads), train_set,
        num_boost_round=rounds, valid_sets=[valid_set], feval=lgb_brier,
        callbacks=[lgb.early_stopping(80 if args.preset == "full" else 20),
                   lgb.log_evaluation(50)],
    )


def cat_params(args, iterations, seed):
    p = dict(
        iterations=iterations, learning_rate=.02, depth=6,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=100.,
        random_strength=1., random_seed=seed, border_count=32,
        allow_writing_files=False, verbose=100,
        task_type=args.task_type, thread_count=args.threads,
    )
    if args.task_type == "GPU": p["devices"] = args.devices
    return p


def fit_cat(x_train, y_train, args, seed):
    rounds = 1200 if args.preset == "full" else 150
    model = CatBoostClassifier(**cat_params(args, rounds, seed))
    # IDs and categories intentionally remain numeric. On this task, official
    # as-of rates carry safer player history than target-derived CTR features.
    # Iteration count is fixed from prior rolling experiments. Avoid passing an
    # eval_set here: CatBoost's Brier metric is CPU-only on GPU training and the
    # validation predictions are scored once with NumPy immediately afterwards.
    model.fit(x_train, y_train)
    return model


def fit_categorical_cat(x_train, y_train, args, seed):
    """Fit an intentionally diverse CatBoost using native categorical IDs."""
    rounds = 1200 if args.preset == "full" else 150
    params = cat_params(args, rounds, seed)
    params.update(max_ctr_complexity=1, one_hot_max_size=32)
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, cat_features=CAT_COLUMNS)
    return model


def fit_categorical_count_expert(x_train, y_train, args, seed, two_strike):
    mask = x_train["two_strike"].to_numpy() == int(two_strike)
    return fit_categorical_cat(x_train.loc[mask], y_train[mask], args, seed)


def fit_brier_regressor(x_train, y_train, args, seed):
    """Optimize squared probability error directly instead of Logloss."""
    rounds = 1200 if args.preset == "full" else 150
    params = cat_params(args, rounds, seed)
    params.update(loss_function="RMSE", eval_metric="RMSE")
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train)
    return model


def fit_count_expert(x_train, y_train, args, seed, two_strike):
    """Fit a CatBoost specialist for one side of the two-strike gate."""
    mask = x_train["two_strike"].to_numpy() == int(two_strike)
    return fit_cat(x_train.loc[mask], y_train[mask], args, seed)


def predict_count_expert(other_model, two_strike_model, x_valid):
    """Route each row to the specialist matching its pre-pitch count."""
    gate = x_valid["two_strike"].to_numpy().astype(bool)
    prediction = np.empty(len(x_valid), dtype=np.float64)
    if (~gate).any():
        prediction[~gate] = other_model.predict_proba(x_valid.loc[~gate])[:, 1]
    if gate.any():
        prediction[gate] = two_strike_model.predict_proba(x_valid.loc[gate])[:, 1]
    return prediction


def optimize_blend(y, years, matrix):
    """Select the latest-fold ensemble without forcing weak models into it."""
    latest = years == np.max(years)
    latest_y = y[latest]
    latest_matrix = matrix[latest]
    reference = float(latest_y.mean() * (1.0 - latest_y.mean()))
    model_count = matrix.shape[1]

    def objective(z):
        w, intercept, slope = z[:model_count], z[model_count], z[model_count + 1]
        pred = np.clip(intercept + slope * (latest_matrix @ w), .005, .995)
        calibration_penalty = .005 * intercept**2 + .002 * (slope - 1.0)**2
        return float(brier(latest_y, pred) / reference + calibration_penalty)

    start = np.array([0., 0., .5, 0., .5, 0., 1.])
    result = minimize(objective, start, method="SLSQP",
                      bounds=[(0,0),(0,0),(0,1),(0,0),(0,1)] + [(-.08,.08),(.75,1.25)],
                      constraints={"type":"eq", "fun":lambda z: z[:model_count].sum()-1},
                      options={"maxiter":500, "ftol":1e-12})
    if not result.success: raise RuntimeError(f"Blend optimization failed: {result.message}")
    return result.x


def optimize_segment_blends(y, years, matrix, two_strike_gate):
    """Fit independent global/specialist blends for each pre-pitch segment."""
    latest_year = np.max(years)
    parameters = {}
    candidate_count = len(SEGMENT_MODEL_INDICES)
    for label, gate_value in (("other", False), ("two_strike", True)):
        mask = (years == latest_year) & (two_strike_gate == gate_value)
        target = y[mask]
        predictions = matrix[mask][:, SEGMENT_MODEL_INDICES]
        reference = float(target.mean() * (1.0 - target.mean()))

        def objective(z):
            pred = np.clip(
                z[candidate_count] + z[candidate_count + 1]
                * (predictions @ z[:candidate_count]), .005, .995,
            )
            penalty = (
                .005 * z[candidate_count]**2
                + .002 * (z[candidate_count + 1] - 1.0)**2
            )
            return float(brier(target, pred) / reference + penalty)

        result = minimize(
            objective, np.r_[np.full(candidate_count, 1/candidate_count), 0., 1.],
            method="SLSQP",
            bounds=[(0,1)] * candidate_count + [(-.08,.08),(.75,1.25)],
            constraints={
                "type":"eq", "fun":lambda z: z[:candidate_count].sum()-1
            },
            options={"maxiter":500, "ftol":1e-12},
        )
        if not result.success:
            raise RuntimeError(f"{label} blend optimization failed: {result.message}")
        parameters[label] = result.x
    return parameters


def apply_segment_blends(matrix, two_strike_gate, parameters):
    prediction = np.empty(len(matrix), dtype=np.float64)
    candidates = matrix[:, SEGMENT_MODEL_INDICES]
    for label, gate_value in (("other", False), ("two_strike", True)):
        mask = two_strike_gate == gate_value
        z = parameters[label]
        prediction[mask] = np.clip(
            z[-2] + z[-1] * (candidates[mask] @ z[:-2]), .005, .995
        )
    return prediction


def main() -> None:
    args = arguments(); started = time.time(); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.data_dir) / "train.csv"
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    y_series = raw.pop(TARGET_COL).astype(np.float32); y = y_series.to_numpy()
    print("Building time-safe features...")
    bases = training_history_arrays(raw, y_series)
    x = engineer_features(raw, *bases, global_prior=float(y.mean()))
    add_training_component_features(x, raw)
    x = add_state_interactions(x)
    for col in CAT_COLUMNS: x[col] = x[col].fillna(-1).astype(np.int32)
    years = raw["season"].to_numpy(np.int16)

    fold_predictions, fold_targets, fold_years, fold_gates = [], [], [], []
    component_reports = {}
    best_lgb = [[], []]
    for valid_year in (2023, 2024):
        tr, va = years < valid_year, years == valid_year
        print(f"Rolling fold: train < {valid_year}, validate {valid_year}")
        predictions = []
        for variant in (0, 1):
            model = fit_lgb(x.loc[tr], y[tr], x.loc[va], y[va],
                            season_weights(raw.loc[tr, "season"], valid_year-1), variant, args)
            predictions.append(model.predict(x.loc[va], num_iteration=model.best_iteration))
            best_lgb[variant].append(model.best_iteration)
        cat_predictions = []
        for seed in (42, 43, 44):
            cat = fit_cat(x.loc[tr], y[tr], args, seed)
            cat_predictions.append(cat.predict_proba(x.loc[va])[:, 1])
        predictions.append(np.mean(cat_predictions, axis=0))
        predictions.append(history_expert(x.loc[va], float(y[tr].mean())))
        expert_predictions = []
        for seed in (52, 53, 54):
            other = fit_count_expert(x.loc[tr], y[tr], args, seed, False)
            two_strike = fit_count_expert(x.loc[tr], y[tr], args, seed, True)
            expert_predictions.append(
                predict_count_expert(other, two_strike, x.loc[va])
            )
        predictions.append(np.mean(expert_predictions, axis=0))
        categorical_predictions = []
        for seed in (62, 63, 64):
            categorical = fit_categorical_cat(x.loc[tr], y[tr], args, seed)
            categorical_predictions.append(categorical.predict_proba(x.loc[va])[:, 1])
        predictions.append(np.mean(categorical_predictions, axis=0))
        categorical_expert_predictions = []
        for seed in (72, 73, 74):
            other = fit_categorical_count_expert(x.loc[tr], y[tr], args, seed, False)
            two_strike = fit_categorical_count_expert(x.loc[tr], y[tr], args, seed, True)
            categorical_expert_predictions.append(
                predict_count_expert(other, two_strike, x.loc[va])
            )
        predictions.append(np.mean(categorical_expert_predictions, axis=0))
        regressor_predictions = []
        for seed in (82, 83, 84):
            regressor = fit_brier_regressor(x.loc[tr], y[tr], args, seed)
            regressor_predictions.append(regressor.predict(x.loc[va]))
        predictions.append(np.mean(regressor_predictions, axis=0))
        matrix = np.column_stack(predictions)
        component_reports[str(valid_year)] = {
            name: {"brier": brier(y[va], matrix[:, i]), "bss": bss(y[va], matrix[:, i])}
            for i, name in enumerate(MODEL_NAMES)
        }
        print(component_reports[str(valid_year)])
        fold_predictions.append(matrix); fold_targets.append(y[va]); fold_years.append(years[va])
        fold_gates.append(x.loc[va, "two_strike"].to_numpy().astype(bool))

    oof = np.vstack(fold_predictions); oof_y = np.concatenate(fold_targets)
    oof_year = np.concatenate(fold_years); oof_gate = np.concatenate(fold_gates)
    segment_parameters = optimize_segment_blends(oof_y, oof_year, oof, oof_gate)
    blended = apply_segment_blends(oof, oof_gate, segment_parameters)
    weights = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
    intercept, slope = 0., 1.
    reports = {}
    for year in (2023, 2024):
        m = oof_year == year
        reports[str(year)] = {"brier": brier(oof_y[m], blended[m]), "bss": bss(oof_y[m], blended[m]),
                              "target_rate": float(oof_y[m].mean()), "prediction_mean": float(blended[m].mean())}
    print("Segment blends:", {k: v.tolist() for k, v in segment_parameters.items()})
    print("OOF:", reports)
    diagnostics = Path(args.diagnostic_dir); diagnostics.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostics / "v11_oof_predictions.npz", predictions=oof,
        target=oof_y, season=oof_year, model_names=np.asarray(MODEL_NAMES),
        two_strike=oof_gate, blended=blended,
    )

    lgb_rounds = [max(50, int(round(np.median(v)*1.05))) for v in best_lgb]
    cat_rounds = 1320 if args.preset == "full" else 170
    final_lgb = []
    all_weights = season_weights(raw["season"], int(years.max()))
    for variant in (0,1):
        ds = lgb.Dataset(x, label=y, weight=all_weights)
        final_lgb.append(lgb.train(lgb_params(variant, args.seed+variant*97, args.threads), ds,
                                   num_boost_round=lgb_rounds[variant]))
        final_lgb[-1].save_model(str(out / f"lgb_{variant}.txt"))
    for index, seed in enumerate((42, 43, 44)):
        final_cat = CatBoostClassifier(**cat_params(args, cat_rounds, seed))
        final_cat.fit(x, y)
        final_cat.save_model(str(out / f"catboost_{index}.cbm"))
    for index, seed in enumerate((52, 53, 54)):
        for label, two_strike in (("other", False), ("two_strike", True)):
            mask = x["two_strike"].to_numpy() == int(two_strike)
            expert = CatBoostClassifier(**cat_params(args, cat_rounds, seed))
            expert.fit(x.loc[mask], y[mask])
            expert.save_model(str(out / f"catboost_{label}_{index}.cbm"))
    for index, seed in enumerate((62, 63, 64)):
        categorical = CatBoostClassifier(**{
            **cat_params(args, cat_rounds, seed),
            "max_ctr_complexity": 1, "one_hot_max_size": 32,
        })
        categorical.fit(x, y, cat_features=CAT_COLUMNS)
        categorical.save_model(str(out / f"catboost_categorical_{index}.cbm"))
    for index, seed in enumerate((72, 73, 74)):
        for label, two_strike in (("other", False), ("two_strike", True)):
            mask = x["two_strike"].to_numpy() == int(two_strike)
            categorical_expert = CatBoostClassifier(**{
                **cat_params(args, cat_rounds, seed),
                "max_ctr_complexity": 1, "one_hot_max_size": 32,
            })
            categorical_expert.fit(x.loc[mask], y[mask], cat_features=CAT_COLUMNS)
            categorical_expert.save_model(
                str(out / f"catboost_categorical_{label}_{index}.cbm")
            )
    for index, seed in enumerate((82, 83, 84)):
        regressor = CatBoostRegressor(**{
            **cat_params(args, cat_rounds, seed),
            "loss_function": "RMSE", "eval_metric": "RMSE",
        })
        regressor.fit(x, y)
        regressor.save_model(str(out / f"catboost_brier_{index}.cbm"))

    metadata = {
        "version":"v11_brier_regression", "feature_columns":x.columns.tolist(),
        "cat_features":[], "history":asdict(build_end_history(raw, y_series)),
        "model_names":MODEL_NAMES, "blend_weights":dict(zip(MODEL_NAMES, weights.tolist())),
        "calibration":{"intercept":intercept, "slope":slope}, "clip":[.005,.995],
        "segment_blends": {
            key: {"weights":dict(zip(SEGMENT_MODEL_NAMES, value[:-2].tolist())),
                  "intercept":float(value[-2]), "slope":float(value[-1])}
            for key, value in segment_parameters.items()
        },
        "training_info":{"lgb_rounds":lgb_rounds, "catboost_rounds":cat_rounds,
                         "component_reports":component_reports,
                         "rolling_reports":reports, "elapsed_seconds":time.time()-started},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"Saved v11 artifacts to {out}; diagnostics={diagnostics}; elapsed={time.time()-started:.1f}s")

if __name__ == "__main__": main()
