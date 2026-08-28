"""Post-break F-regime command models for strict 2023 -> 2024 validation.

The F target and reconstructed command components have a documented structural
break after 2022.  This screen therefore trains only on 2023 F rows and applies
the models only to 2024 F rows.  Evaluation inputs remain row-independent.
"""
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
CLASS_NAMES = (
    "success", "reverse_only", "middle_only", "reverse_middle", "wayoff",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed, multiclass=False):
    loss = "MultiClass" if multiclass else "Logloss"
    result = dict(
        iterations=args.iterations, learning_rate=.025, depth=6,
        l2_leaf_reg=160., random_strength=2., bagging_temperature=.8,
        border_count=32, bootstrap_type="Bayesian", loss_function=loss,
        eval_metric=loss, random_seed=4800 + seed, task_type=args.task_type,
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
    wayoff = labels["wayoff"].fillna(0).eq(1).to_numpy()
    overlap = (reverse & middle).astype(np.int8)

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
    f_rows = raw["game_type"].eq("F").to_numpy()
    train = (seasons == 2023) & f_rows & complete
    valid = seasons == 2024
    valid_f = f_rows[valid]
    print(json.dumps({
        "train_rows": int(train.sum()), "valid_F_rows": int(valid_f.sum()),
        "features": int(features.shape[1]),
        "train_success_rate": float(target[train].mean()),
        "valid_success_rate_audit_only": float(target[valid & f_rows].mean()),
        "train_overlap_rate": float(overlap[train].mean()),
    }), flush=True)

    predictions = {}
    direct = CatBoostClassifier(**parameters(args, 0))
    direct.fit(
        features.loc[train], target[train],
        cat_features=list(LOW_CARD_CATEGORIES),
    )
    predictions["direct"] = direct.predict_proba(features.loc[valid])[:, 1]

    multiclass = CatBoostClassifier(**parameters(args, 1, multiclass=True))
    multiclass.fit(
        features.loc[train], outcome[train],
        cat_features=list(LOW_CARD_CATEGORIES),
    )
    probabilities = multiclass.predict_proba(features.loc[valid])
    success_position = int(np.flatnonzero(
        np.asarray(multiclass.classes_, dtype=int) == 0
    )[0])
    predictions["multiclass"] = probabilities[:, success_position]

    components = {}
    for offset, (name, label) in enumerate((
        ("reverse", reverse), ("middle", middle), ("wayoff", wayoff),
        ("overlap", overlap),
    ), 2):
        model = CatBoostClassifier(**parameters(args, offset))
        model.fit(
            features.loc[train], np.asarray(label[train], dtype=np.int8),
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        components[name] = model.predict_proba(features.loc[valid])[:, 1]
        print(f"completed component={name}", flush=True)
    predictions["union"] = np.clip(
        1. - components["reverse"] - components["middle"]
        - components["wayoff"] + components["overlap"], .005, .995,
    )

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        active = archive["season"] == 2024
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v38 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    base_logit = logit(base)
    directions = {
        name: logit(np.clip(value, .005, .995)) - base_logit
        for name, value in predictions.items()
    }

    individual = []
    for name, direction in directions.items():
        for weight in np.round(np.arange(-.10, .401, .025), 3):
            candidate = base.copy()
            candidate[valid_f] = sigmoid(
                base_logit[valid_f] + weight * direction[valid_f]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {key: result[key] - baseline[key] for key in result}
            individual.append({
                "name": name, "weight": float(weight), "scores": result,
                "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })

    # The direct and multiclass outputs share a target, while the coherent
    # union is structurally different.  A small grid checks complementarity.
    combinations = []
    grid = np.round(np.arange(0., .201, .025), 3)
    for direct_weight in grid:
        for multi_weight in grid:
            for union_weight in grid:
                candidate = base.copy()
                candidate[valid_f] = sigmoid(
                    base_logit[valid_f]
                    + direct_weight * directions["direct"][valid_f]
                    + multi_weight * directions["multiclass"][valid_f]
                    + union_weight * directions["union"][valid_f]
                )
                result = score(fold_target, candidate, blocks, game_type)
                gains = {key: result[key] - baseline[key] for key in result}
                combinations.append({
                    "direct_weight": float(direct_weight),
                    "multi_weight": float(multi_weight),
                    "union_weight": float(union_weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(
                        gains[f"q{i}"] for i in range(1, 5)
                    )),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })

    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["F"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "baseline": baseline,
        "standalone": {
            name: {
                "F_bss": float(bss(
                    fold_target[valid_f], prediction[valid_f]
                )),
                "F_mean": float(prediction[valid_f].mean()),
                "correlation_base_F": float(np.corrcoef(
                    prediction[valid_f], base[valid_f]
                )[0, 1]),
            } for name, prediction in predictions.items()
        },
        "direction_correlations_F": {
            f"{left}_{right}": float(np.corrcoef(
                directions[left][valid_f], directions[right][valid_f]
            )[0, 1])
            for index, left in enumerate(directions)
            for right in list(directions)[index + 1:]
        },
        "best_individual_robust": sorted(
            individual, key=robust_key, reverse=True,
        )[:40],
        "best_individual_score": sorted(
            individual, key=lambda row: row["scores"]["all"], reverse=True,
        )[:40],
        "best_combination_robust": sorted(
            combinations, key=robust_key, reverse=True,
        )[:60],
        "best_combination_score": sorted(
            combinations, key=lambda row: row["scores"]["all"], reverse=True,
        )[:60],
    }
    output = ROOT / "research/v48_regime_command_2024.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
        **{f"component_{name}": value.astype(np.float32)
           for name, value in components.items()},
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
