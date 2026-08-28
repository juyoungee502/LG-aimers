"""Train low-cardinality direct models with prior-season failure context.

Two objectives are screened from the same chronological feature matrix:
binary Logloss and RMSE, the latter directly matching Brier-error geometry.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.errors import PerformanceWarning

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss, reconstruct_labels
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--objectives", default="logloss,rmse")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, objective):
    result = dict(
        iterations=args.iterations,
        learning_rate=.016,
        depth=8,
        l2_leaf_reg=450.,
        random_strength=2.8,
        bagging_temperature=.5,
        border_count=32,
        bootstrap_type="Bayesian",
        loss_function="Logloss" if objective == "logloss" else "RMSE",
        eval_metric="Logloss" if objective == "logloss" else "RMSE",
        task_type=args.task_type,
        thread_count=args.threads,
        random_seed=4700 if objective == "logloss" else 4701,
        allow_writing_files=False,
        verbose=100,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def score(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    regular = game_type == "R"
    result["R"] = float(bss(target[regular], prediction[regular]))
    result["F"] = float(bss(target[~regular], prediction[~regular]))
    return result


def main():
    args = arguments()
    objectives = tuple(item.strip() for item in args.objectives.split(",") if item.strip())
    unknown = sorted(set(objectives) - {"logloss", "rmse"})
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])
    for column in LOW_CARD_CATEGORIES:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    print(json.dumps({
        "valid_year": args.valid_year,
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "features": int(features.shape[1]),
        "objectives": objectives,
    }), flush=True)

    predictions = {}
    for objective in objectives:
        print(f"Training v47 context-direct objective={objective}", flush=True)
        cls = CatBoostClassifier if objective == "logloss" else CatBoostRegressor
        model = cls(**parameters(args, objective))
        model.fit(
            features.loc[train], target[train], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        if objective == "logloss":
            prediction = model.predict_proba(features.loc[valid])[:, 1]
        else:
            prediction = model.predict(features.loc[valid])
        predictions[objective] = np.clip(prediction, .005, .995)

    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        v24 = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v24 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    if args.valid_year == 2024:
        with np.load(
            ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
        ) as archive:
            failure = archive["new_failure"].astype(float)
        with np.load(
            ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz", allow_pickle=True,
        ) as archive:
            direct = archive["prediction"].astype(float)
        first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
        base = sigmoid(.90 * logit(first) + .10 * logit(direct))
    else:
        base = v24
    blocks = masks(len(fold_target))
    baseline = score(fold_target, base, blocks, game_type)

    reports = []
    for objective, prediction in predictions.items():
        direction = logit(prediction) - logit(base)
        for gate in ("all", "R", "F"):
            selected = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(-.10, .301, .025), 3):
                candidate = base.copy()
                candidate[selected] = sigmoid(
                    logit(base[selected]) + weight * direction[selected]
                )
                result = score(fold_target, candidate, blocks, game_type)
                gains = {name: result[name] - baseline[name] for name in result}
                reports.append({
                    "objective": objective, "gate": gate, "weight": float(weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })
    robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"],
                row["gains"]["R"], row["gains"]["F"]),
            row["scores"]["all"],
        ), reverse=True,
    )
    by_score = sorted(reports, key=lambda row: row["scores"]["all"], reverse=True)
    diagnostics = {
        "valid_year": args.valid_year,
        "baseline": baseline,
        "models": {
            objective: {
                "bss": float(bss(fold_target, prediction)),
                "mean": float(prediction.mean()),
                "correlation_base": float(np.corrcoef(prediction, base)[0, 1]),
            } for objective, prediction in predictions.items()
        },
        "best_robust": robust[:40],
        "best_score": by_score[:40],
    }
    output = ROOT / "research" / f"v47_context_direct_hl{args.half_life:g}_{args.valid_year}.npz"
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
