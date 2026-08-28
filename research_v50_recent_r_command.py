"""Previous-season-only R direct and coherent command models."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed, multiclass=False):
    loss = "MultiClass" if multiclass else "Logloss"
    result = dict(
        iterations=args.iterations, learning_rate=.02, depth=6,
        l2_leaf_reg=220., random_strength=2., bagging_temperature=.8,
        border_count=32, bootstrap_type="Bayesian", loss_function=loss,
        eval_metric=loss, random_seed=5000 + seed, task_type=args.task_type,
        thread_count=args.threads, allow_writing_files=False, verbose=100,
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
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    complete = labels[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    reverse = labels["reverse"].fillna(0).eq(1).to_numpy()
    middle = labels["middle"].fillna(0).eq(1).to_numpy()
    outcome = np.full(len(raw), -1, dtype=np.int8)
    outcome[complete & (target == 1)] = 0
    outcome[complete & (target == 0) & reverse & ~middle] = 1
    outcome[complete & (target == 0) & ~reverse & middle] = 2
    outcome[complete & (target == 0) & reverse & middle] = 3
    outcome[complete & (target == 0) & ~reverse & ~middle] = 4

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
    r_rows = raw["game_type"].eq("R").to_numpy()
    train = (seasons == args.valid_year - 1) & r_rows & complete
    valid = seasons == args.valid_year
    valid_r = r_rows[valid]
    print(json.dumps({
        "valid_year": args.valid_year, "n_seeds": args.n_seeds,
        "train_rows": int(train.sum()), "valid_R_rows": int(valid_r.sum()),
        "features": int(features.shape[1]),
        "train_rate": float(target[train].mean()),
        "valid_rate_audit_only": float(target[valid & r_rows].mean()),
    }), flush=True)

    predictions = {"direct": [], "multiclass": []}
    for seed_index in range(args.n_seeds):
        direct = CatBoostClassifier(**parameters(args, 101 * seed_index))
        direct.fit(
            features.loc[train], target[train],
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        predictions["direct"].append(
            direct.predict_proba(features.loc[valid])[:, 1]
        )

        multiclass = CatBoostClassifier(**parameters(
            args, 1 + 101 * seed_index, multiclass=True,
        ))
        multiclass.fit(
            features.loc[train], outcome[train],
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        probabilities = multiclass.predict_proba(features.loc[valid])
        success_position = int(np.flatnonzero(
            np.asarray(multiclass.classes_, dtype=int) == 0
        )[0])
        predictions["multiclass"].append(probabilities[:, success_position])
        print(f"completed seed={seed_index + 1}/{args.n_seeds}", flush=True)
    predictions = {
        name: np.mean(values, axis=0) for name, values in predictions.items()
    }

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v38 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    directions = {
        name: logit(np.clip(value, .005, .995)) - logit(base)
        for name, value in predictions.items()
    }
    reports = []
    grid = np.round(np.arange(-.10, .301, .025), 3)
    for name, direction in directions.items():
        for weight in grid:
            candidate = base.copy()
            candidate[valid_r] = sigmoid(
                logit(base[valid_r]) + weight * direction[valid_r]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {key: result[key] - baseline[key] for key in result}
            reports.append({
                "name": name, "weight": float(weight), "scores": result,
                "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })

    combinations = []
    combo_grid = np.round(np.arange(-.10, .201, .025), 3)
    for direct_weight in combo_grid:
        for multi_weight in combo_grid:
            candidate = base.copy()
            candidate[valid_r] = sigmoid(
                logit(base[valid_r])
                + direct_weight * directions["direct"][valid_r]
                + multi_weight * directions["multiclass"][valid_r]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {key: result[key] - baseline[key] for key in result}
            combinations.append({
                "direct_weight": float(direct_weight),
                "multi_weight": float(multi_weight),
                "scores": result, "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["R"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "baseline": baseline,
        "standalone": {
            name: {
                "R_bss": float(bss(fold_target[valid_r], value[valid_r])),
                "R_mean": float(value[valid_r].mean()),
                "correlation_base_R": float(np.corrcoef(
                    value[valid_r], base[valid_r]
                )[0, 1]),
            } for name, value in predictions.items()
        },
        "direction_correlation_R": float(np.corrcoef(
            directions["direct"][valid_r], directions["multiclass"][valid_r]
        )[0, 1]),
        "best_individual_robust": sorted(
            reports, key=robust_key, reverse=True,
        )[:40],
        "best_individual_score": sorted(
            reports, key=lambda row: row["scores"]["all"], reverse=True,
        )[:40],
        "best_combination_robust": sorted(
            combinations, key=robust_key, reverse=True,
        )[:50],
        "best_combination_score": sorted(
            combinations, key=lambda row: row["scores"]["all"], reverse=True,
        )[:50],
    }
    output = ROOT / f"research/v50_recent_r_command_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
