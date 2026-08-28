"""Correct double-counted reverse/middle failures by inclusion-exclusion."""
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
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args):
    result = dict(
        iterations=args.iterations,
        learning_rate=.01631820635235777,
        depth=8,
        l2_leaf_reg=509.6419153575998,
        random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515,
        border_count=32,
        bootstrap_type="Bayesian",
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type=args.task_type,
        thread_count=args.threads,
        random_seed=4500,
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
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    full = pd.concat([raw, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    complete = labels[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    overlap = (
        labels["reverse"].fillna(0).eq(1)
        & labels["middle"].fillna(0).eq(1)
    ).to_numpy(np.int8)

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
    train = (seasons < args.valid_year) & complete
    valid = seasons == args.valid_year
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    print(json.dumps({
        "valid_year": args.valid_year,
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "train_overlap_rate": float(overlap[train].mean()),
        "valid_overlap_rate_observed_for_audit_only": float(
            overlap[valid & complete].mean()
        ),
        "features": int(features.shape[1]),
    }), flush=True)
    model = CatBoostClassifier(**parameters(args))
    model.fit(
        features.loc[train], overlap[train],
        sample_weight=sample_weight,
        cat_features=list(LOW_CARD_CATEGORIES),
    )
    overlap_probability = model.predict_proba(features.loc[valid])[:, 1]

    with np.load(
        ROOT / f"research/v34_categorical_failure_lowcard_no_ids_hl2_{args.valid_year}.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
        fold_target = archive["target"].astype(float)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        v24 = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v34 rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(fold_target))

    standalone = []
    reports = []
    direct = None
    if args.valid_year == 2024:
        with np.load(
            ROOT / "research/v35_lowcard_direct_hl2_s3_2024.npz",
            allow_pickle=True,
        ) as archive:
            direct = archive["prediction"].astype(float)
        first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
        base = sigmoid(.90 * logit(first) + .10 * logit(direct))
    else:
        base = v24
    baseline = score(fold_target, base, blocks, game_type)

    for inclusion_scale in np.round(np.arange(0., 1.501, .05), 3):
        corrected = np.clip(
            failure + inclusion_scale * overlap_probability, .005, .995,
        )
        standalone.append({
            "inclusion_scale": float(inclusion_scale),
            "bss": float(bss(fold_target, corrected)),
            "mean": float(corrected.mean()),
        })
        for gate in ("all", "R", "F"):
            selected = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(0., .301, .025), 3):
                candidate = base.copy()
                candidate[selected] = sigmoid(
                    logit(base[selected]) + weight * (
                        logit(corrected[selected]) - logit(base[selected])
                    )
                )
                result = score(fold_target, candidate, blocks, game_type)
                gains = {name: result[name] - baseline[name] for name in result}
                reports.append({
                    "inclusion_scale": float(inclusion_scale),
                    "gate": gate, "weight": float(weight),
                    "scores": result, "gains": gains,
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                })
    robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"],
                row["gains"]["R"], row["gains"]["F"]),
            row["gains"]["all"],
        ), reverse=True,
    )
    by_score = sorted(reports, key=lambda row: row["scores"]["all"], reverse=True)
    diagnostics = {
        "valid_year": args.valid_year,
        "overlap_prediction_mean": float(overlap_probability.mean()),
        "baseline": baseline,
        "raw_failure_bss": float(bss(fold_target, failure)),
        "best_standalone": max(standalone, key=lambda row: row["bss"]),
        "best_robust": robust[:40],
        "best_score": by_score[:40],
    }
    output = ROOT / "research" / f"v45_overlap_hl{args.half_life:g}_{args.valid_year}.npz"
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        overlap_probability=overlap_probability.astype(np.float32),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
