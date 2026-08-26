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
from catboost import CatBoostClassifier
from scipy.optimize import minimize

from feature_engineering import (
    ID_COL, TARGET_COL, build_end_history, engineer_features,
    training_history_arrays,
)

CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
MODEL_NAMES = ["lgb_a", "lgb_b", "catboost", "history_expert"]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="submit/model")
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


def cat_params(args, iterations):
    p = dict(
        iterations=iterations, learning_rate=.035, depth=7,
        loss_function="Logloss", eval_metric="BrierScore", l2_leaf_reg=14.,
        random_strength=.5, random_seed=args.seed + 701, border_count=96,
        one_hot_max_size=16, allow_writing_files=False, verbose=50,
        task_type=args.task_type, thread_count=args.threads,
    )
    if args.task_type == "GPU": p["devices"] = args.devices
    return p


def fit_cat(x_train, y_train, x_valid, y_valid, weights, args):
    rounds = 700 if args.preset == "full" else 100
    model = CatBoostClassifier(**cat_params(args, rounds))
    model.fit(x_train, y_train, cat_features=CAT_COLUMNS, sample_weight=weights,
              eval_set=(x_valid, y_valid), early_stopping_rounds=80, use_best_model=True)
    return model


def optimize_blend(y, years, matrix):
    """Minimize mean yearly normalized Brier, not pooled row-count-weighted loss."""
    unique_years = np.unique(years)
    def objective(z):
        w, intercept, slope = z[:4], z[4], z[5]
        pred = np.clip(intercept + slope * (matrix @ w), .005, .995)
        losses = []
        for year in unique_years:
            m = years == year; r = float(y[m].mean())
            losses.append(brier(y[m], pred[m]) / (r * (1-r)))
        penalty = .002 * np.sum((w - .25) ** 2) + .02 * intercept**2 + .01 * (slope-1)**2
        return float(np.mean(losses) + penalty)
    start = np.array([.30, .25, .25, .20, 0., 1.])
    result = minimize(objective, start, method="SLSQP",
                      bounds=[(0,1)]*4 + [(-.08,.08),(.75,1.25)],
                      constraints={"type":"eq", "fun":lambda z: z[:4].sum()-1},
                      options={"maxiter":500, "ftol":1e-12})
    if not result.success: raise RuntimeError(f"Blend optimization failed: {result.message}")
    z = result.x
    # Conservative calibration shrinkage protects against 2025 regime shift.
    z[4] *= .75; z[5] = 1 + .75 * (z[5]-1)
    return z


def main() -> None:
    args = arguments(); started = time.time(); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.data_dir) / "train.csv"
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    y_series = raw.pop(TARGET_COL).astype(np.float32); y = y_series.to_numpy()
    print("Building time-safe features...")
    bases = training_history_arrays(raw, y_series)
    x = engineer_features(raw, *bases, global_prior=float(y.mean()))
    for col in CAT_COLUMNS: x[col] = x[col].fillna(-1).astype(np.int32)
    years = raw["season"].to_numpy(np.int16)

    fold_predictions, fold_targets, fold_years = [], [], []
    best_lgb = [[], []]; best_cat = []
    for valid_year in (2023, 2024):
        tr, va = years < valid_year, years == valid_year
        print(f"Rolling fold: train < {valid_year}, validate {valid_year}")
        predictions = []
        for variant in (0, 1):
            model = fit_lgb(x.loc[tr], y[tr], x.loc[va], y[va],
                            season_weights(raw.loc[tr, "season"], valid_year-1), variant, args)
            predictions.append(model.predict(x.loc[va], num_iteration=model.best_iteration))
            best_lgb[variant].append(model.best_iteration)
        cat = fit_cat(x.loc[tr], y[tr], x.loc[va], y[va],
                      season_weights(raw.loc[tr, "season"], valid_year-1), args)
        predictions.append(cat.predict_proba(x.loc[va])[:, 1])
        best_cat.append(cat.get_best_iteration()+1)
        predictions.append(history_expert(x.loc[va], float(y[tr].mean())))
        matrix = np.column_stack(predictions)
        print({name: bss(y[va], matrix[:, i]) for i, name in enumerate(MODEL_NAMES)})
        fold_predictions.append(matrix); fold_targets.append(y[va]); fold_years.append(years[va])

    oof = np.vstack(fold_predictions); oof_y = np.concatenate(fold_targets)
    oof_year = np.concatenate(fold_years); z = optimize_blend(oof_y, oof_year, oof)
    weights, intercept, slope = z[:4], float(z[4]), float(z[5])
    blended = np.clip(intercept + slope * (oof @ weights), .005, .995)
    reports = {}
    for year in (2023, 2024):
        m = oof_year == year
        reports[str(year)] = {"brier": brier(oof_y[m], blended[m]), "bss": bss(oof_y[m], blended[m]),
                              "target_rate": float(oof_y[m].mean()), "prediction_mean": float(blended[m].mean())}
    print("Blend weights:", dict(zip(MODEL_NAMES, weights))); print("OOF:", reports)

    lgb_rounds = [max(50, int(round(np.median(v)*1.05))) for v in best_lgb]
    cat_rounds = max(50, int(round(np.median(best_cat)*1.05)))
    final_lgb = []
    all_weights = season_weights(raw["season"], int(years.max()))
    for variant in (0,1):
        ds = lgb.Dataset(x, label=y, weight=all_weights)
        final_lgb.append(lgb.train(lgb_params(variant, args.seed+variant*97, args.threads), ds,
                                   num_boost_round=lgb_rounds[variant]))
        final_lgb[-1].save_model(str(out / f"lgb_{variant}.txt"))
    final_cat = CatBoostClassifier(**cat_params(args, cat_rounds))
    final_cat.fit(x, y, cat_features=CAT_COLUMNS, sample_weight=all_weights)
    final_cat.save_model(str(out / "catboost.cbm"))

    metadata = {
        "version":"v3_rolling_bss_ensemble", "feature_columns":x.columns.tolist(),
        "cat_features":CAT_COLUMNS, "history":asdict(build_end_history(raw, y_series)),
        "model_names":MODEL_NAMES, "blend_weights":dict(zip(MODEL_NAMES, weights.tolist())),
        "calibration":{"intercept":intercept, "slope":slope}, "clip":[.005,.995],
        "training_info":{"lgb_rounds":lgb_rounds, "catboost_rounds":cat_rounds,
                         "rolling_reports":reports, "elapsed_seconds":time.time()-started},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"Saved v3 artifacts to {out}; elapsed={time.time()-started:.1f}s")

if __name__ == "__main__": main()
