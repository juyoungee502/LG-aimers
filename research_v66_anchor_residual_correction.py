"""Test the public 1146-style small residual correction over the v64 anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from research_v66_hierarchical_residual import pitcher_bootstrap, report_segments
from v66_hierarchical_residual import (
    CLIP, RESIDUAL_CATEGORICAL, TARGET_COL,
    build_anchor_residual_features,
)


ROOT = Path(__file__).resolve().parent
SCALES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
SPECS = (
    {"name": "public_minimal_d3", "feature_set": "minimal", "depth": 3,
     "iterations": 180, "l2": 100.0, "seed": 6681},
    {"name": "structural_d3", "feature_set": "structural", "depth": 3,
     "iterations": 220, "l2": 100.0, "seed": 6682},
    {"name": "structural_d4", "feature_set": "structural", "depth": 4,
     "iterations": 220, "l2": 200.0, "seed": 6683},
)
MINIMAL = (
    "anchor", "hierarchical_prediction", "prediction_gap",
    "absolute_prediction_gap", "squared_prediction_gap",
    "prediction_midpoint", "anchor_uncertainty",
    "pitcher_batter_career_gap", "recent_success_mean",
    "recent_success_std", "log_pitcher_n", "log_batter_n",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def parameters(args: argparse.Namespace, spec: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "iterations": int(spec["iterations"]),
        "depth": int(spec["depth"]),
        "learning_rate": 0.03,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": float(spec["l2"]),
        "random_strength": 0.25,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "one_hot_max_size": 16,
        "random_seed": int(spec["seed"]),
        "task_type": args.task_type,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if args.task_type == "GPU":
        result.update(devices=args.devices, border_count=32, gpu_ram_part=0.85)
    return result


def select_features(
    features: pd.DataFrame, feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    if feature_set == "minimal":
        columns = list(MINIMAL)
    elif feature_set == "structural":
        columns = list(features.columns)
    else:
        raise ValueError(feature_set)
    selected = features[columns]
    categorical = [column for column in RESIDUAL_CATEGORICAL if column in selected]
    return selected, categorical


def fit_fold(
    features: pd.DataFrame,
    target: np.ndarray,
    anchor: np.ndarray,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    spec: dict[str, object],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    matrix, categorical = select_features(features, str(spec["feature_set"]))
    residual = target[fit_index] - anchor[fit_index]
    residual_center = float(residual.mean())
    model = CatBoostRegressor(**parameters(args, spec))
    model.fit(
        matrix.iloc[fit_index], residual - residual_center,
        cat_features=categorical,
    )
    train_prediction = model.predict(matrix.iloc[fit_index])
    prediction_center = float(train_prediction.mean())
    raw = model.predict(matrix.iloc[valid_index])
    corrections = {
        "centered": raw - prediction_center,
        "level": raw + residual_center,
    }
    return corrections, {
        "fit_rows": int(len(fit_index)),
        "valid_rows": int(len(valid_index)),
        "residual_center": residual_center,
        "prediction_center": prediction_center,
        "feature_count": int(matrix.shape[1]),
    }


def main() -> None:
    args = arguments()
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        target = archive["target"].astype(float)
        season = archive["season"].astype(int)
        anchor = archive["blended"].astype(float)
    with np.load(ROOT / "outputs/v66_hierarchical_residual_oof.npz") as archive:
        ours = archive["alternative"].astype(float)
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(target) or not np.array_equal(
        rows[TARGET_COL].to_numpy(float), target,
    ):
        raise ValueError("OOF predictions do not align with train.csv")
    features = build_anchor_residual_features(rows, anchor, ours)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    positions23 = np.flatnonzero((season == 2023) & regular)
    positions24 = np.flatnonzero((season == 2024) & regular)
    first23, second23 = np.array_split(positions23, 2)
    folds = (
        ("2023_h1_to_h2", first23, second23),
        ("2023_to_2024", positions23, positions24),
    )

    candidates: list[dict[str, object]] = []
    fold_predictions: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for spec in SPECS:
        fit_audits: dict[str, object] = {}
        for fold_name, fit_index, valid_index in folds:
            corrections, audit = fit_fold(
                features, target, anchor, fit_index, valid_index, spec, args,
            )
            fit_audits[fold_name] = audit
            for centering, correction in corrections.items():
                fold_predictions[(str(spec["name"]), centering, fold_name)] = (
                    valid_index, correction,
                )
        for centering in ("centered", "level"):
            for scale in SCALES:
                evaluations: dict[str, object] = {}
                bootstrap: dict[str, object] = {}
                for fold_number, (fold_name, _, _) in enumerate(folds):
                    valid_index, correction = fold_predictions[
                        (str(spec["name"]), centering, fold_name)
                    ]
                    candidate = np.clip(
                        anchor[valid_index] + scale * correction, *CLIP,
                    )
                    local_regular = np.ones(len(valid_index), dtype=bool)
                    evaluations[fold_name] = report_segments(
                        target[valid_index], anchor[valid_index], candidate,
                        local_regular,
                    )
                    bootstrap[fold_name] = pitcher_bootstrap(
                        target[valid_index], anchor[valid_index], candidate,
                        rows.iloc[valid_index]["pitcher_id"].to_numpy(),
                        args.bootstrap, 668000 + fold_number,
                    )
                gains = [float(value["gain"]) for value in evaluations.values()]
                candidates.append({
                    "spec": str(spec["name"]),
                    "centering": centering,
                    "scale": scale,
                    "evaluations": evaluations,
                    "bootstrap": bootstrap,
                    "minimum_gain": min(gains),
                    "mean_gain": float(np.mean(gains)),
                    "minimum_positive_probability": min(
                        float(value["positive_probability"])
                        for value in bootstrap.values()
                    ),
                    "fit_audits": fit_audits,
                })
    ranked = sorted(candidates, key=lambda item: (
        item["minimum_gain"], item["minimum_positive_probability"],
        item["mean_gain"],
    ), reverse=True)
    selected = ranked[0]
    strict_gate = bool(
        float(selected["minimum_gain"]) > 0.0
        and float(selected["minimum_positive_probability"]) >= 0.80
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "candidate": "public_1146_style_anchor_residual_correction",
        "selected": selected,
        "strict_gate": strict_gate,
        "top_candidates": ranked[:12],
        "rules": {
            "official_train_only_for_fit": True,
            "external_model_or_prediction_used_for_fit": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research/v66_anchor_residual_correction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
