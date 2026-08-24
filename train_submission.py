"""Train a time-aware LightGBM ensemble and create the submission artifacts.

This is a classical gradient-boosted tree pipeline, not a deep-learning model.
"""

from __future__ import annotations

import argparse
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_engineering import (
    ID_COL,
    TARGET_COL,
    build_end_history,
    engineer_features,
    training_history_arrays,
)


def brier_report(y: np.ndarray, pred: np.ndarray, label: str) -> dict[str, float]:
    r = float(np.mean(y))
    brier = float(np.mean((pred - y) ** 2))
    reference = r * (1.0 - r)
    score = max(0.0, 100000.0 * (1.0 - brier / reference))
    result = {
        "label": label,
        "brier": brier,
        "score": score,
        "target_mean": r,
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
    }
    print(result)
    return result


def model_params(seed: int, variant: int, n_estimators: int) -> dict:
    if variant == 0:
        return dict(
            objective="binary", n_estimators=n_estimators, learning_rate=0.025,
            num_leaves=31, max_depth=-1, min_child_samples=800,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.82,
            reg_alpha=1.0, reg_lambda=12.0, max_bin=127,
            n_jobs=6, verbosity=-1, force_col_wise=True, random_state=seed,
        )
    return dict(
        objective="binary", n_estimators=n_estimators, learning_rate=0.022,
        num_leaves=47, max_depth=-1, min_child_samples=1200,
        subsample=0.80, subsample_freq=1, colsample_bytree=0.72,
        reg_alpha=2.0, reg_lambda=18.0, max_bin=127,
        n_jobs=6, verbosity=-1, force_col_wise=True, random_state=seed,
    )


def make_weights(season: pd.Series) -> np.ndarray:
    # Recent seasons more closely resemble the hidden 2025 evaluation season.
    year_weight = {2019: 0.45, 2020: 0.55, 2021: 0.65,
                   2022: 0.78, 2023: 0.90, 2024: 1.00}
    return season.map(year_weight).fillna(1.0).to_numpy(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./submit/model")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--trees", type=int, default=550)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.csv")
    test_header = pd.read_csv(os.path.join(args.data_dir, "test.csv"), nrows=0)
    feature_cols = [c for c in test_header.columns if c != ID_COL]

    print("Loading train.csv...")
    started = time.time()
    train = pd.read_csv(train_path, usecols=feature_cols + [TARGET_COL])
    target = train[TARGET_COL].astype(np.float32)
    raw = train.drop(columns=[TARGET_COL])
    print(f"Loaded {len(train):,} rows in {time.time() - started:.1f}s")

    print("Building time-safe features...")
    bases = training_history_arrays(raw, target)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    print("Feature matrix:", features.shape)

    is_valid = raw["season"].to_numpy() == 2024
    train_mask = ~is_valid
    valid_y = target.to_numpy()[is_valid]
    valid_predictions = []
    validation_reports = []
    best_iterations = []

    for variant, seed in enumerate((42, 2026)):
        model = lgb.LGBMClassifier(**model_params(seed, variant, args.trees))
        weights = make_weights(raw.loc[train_mask, "season"])
        print(f"Training validation model {variant + 1}/2...")
        model.fit(
            features.loc[train_mask], target.loc[train_mask], sample_weight=weights,
            eval_set=[(features.loc[is_valid], target.loc[is_valid])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(70, verbose=True), lgb.log_evaluation(100)],
        )
        pred = model.predict_proba(
            features.loc[is_valid], num_iteration=model.best_iteration_
        )[:, 1]
        valid_predictions.append(pred)
        model_report = brier_report(valid_y, pred, f"lgb_{variant}")
        model_report["best_iteration"] = int(model.best_iteration_)
        validation_reports.append(model_report)
        best_iterations.append(int(model.best_iteration_))

    blend = 0.55 * valid_predictions[0] + 0.45 * valid_predictions[1]
    blend_report = brier_report(valid_y, blend, "blend_raw")

    # Diagnostic shrink/shift selected from the 2024 holdout.  The conservative
    # factor prevents a single validation year from imposing the full oracle shift.
    matrix = np.column_stack([np.ones(len(blend)), blend])
    intercept, slope = np.linalg.lstsq(matrix, valid_y, rcond=None)[0]
    applied_intercept = float(intercept * 0.50)
    applied_slope = float(1.0 + (slope - 1.0) * 0.50)
    calibrated = np.clip(applied_intercept + applied_slope * blend, 0.01, 0.99)
    calibrated_report = brier_report(valid_y, calibrated, "blend_conservative_calibration")
    print("Calibration:", applied_intercept, applied_slope)

    report = {
        "models": validation_reports,
        "blend": blend_report,
        "calibrated_blend": calibrated_report,
        "calibration_intercept": applied_intercept,
        "calibration_slope": applied_slope,
    }
    os.makedirs("./outputs", exist_ok=True)
    pd.DataFrame(validation_reports + [blend_report, calibrated_report]).to_csv(
        "./outputs/validation_v1.csv", index=False
    )
    joblib.dump(report, "./outputs/validation_v1.pkl", compress=3)
    np.save("./outputs/validation_lgb0_pred.npy", valid_predictions[0])
    np.save("./outputs/validation_lgb1_pred.npy", valid_predictions[1])
    np.save("./outputs/validation_target.npy", valid_y)

    if args.validation_only:
        return

    print("Training final models on 2019-2024...")
    final_models = []
    final_weights = make_weights(raw["season"])
    for variant, seed in enumerate((42, 2026)):
        # Use each validation model's best iteration with a small allowance for
        # the additional 2024 training data.
        best_iteration = best_iterations[variant]
        # A small allowance accounts for the additional 2024 training rows.
        final_trees = max(120, int(round(best_iteration * 1.08)))
        final_model = lgb.LGBMClassifier(**model_params(seed, variant, final_trees))
        final_model.fit(features, target, sample_weight=final_weights)
        final_models.append(final_model)

    history = build_end_history(raw, target)
    bundle = {
        "version": "v1_season_reconstruction_lgbm",
        "models": final_models,
        "blend_weights": [0.55, 0.45],
        "calibration_intercept": applied_intercept,
        "calibration_slope": applied_slope,
        "history": history,
        "feature_columns": list(features.columns),
        "clip": [0.01, 0.99],
        "validation": report,
    }
    os.makedirs(args.model_dir, exist_ok=True)
    output_path = os.path.join(args.model_dir, "model_bundle.pkl")
    joblib.dump(bundle, output_path, compress=3)
    print("Saved:", output_path, os.path.getsize(output_path) / 1024**2, "MiB")


if __name__ == "__main__":
    main()
