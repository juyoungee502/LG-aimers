"""Audit a recent-season overlap correction on regular-season rows.

Reverse and middle failures are added by the v38 failure specialist.  Their
intersection is therefore counted twice.  This experiment estimates only that
intersection from the immediately preceding regular season and adds it back by
inclusion-exclusion before blending with v38.
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
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def parameters(args, seed):
    result = dict(
        iterations=args.iterations, learning_rate=.02, depth=7,
        l2_leaf_reg=260., random_strength=2.4, bagging_temperature=.6,
        border_count=32, bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", random_seed=5100 + seed,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
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
    r_rows = raw["game_type"].eq("R").to_numpy()
    train = (seasons == args.valid_year - 1) & r_rows & complete
    valid = seasons == args.valid_year
    valid_r = r_rows[valid]
    print(json.dumps({
        "valid_year": args.valid_year, "n_seeds": args.n_seeds,
        "train_rows": int(train.sum()), "valid_R_rows": int(valid_r.sum()),
        "train_overlap_rate": float(overlap[train].mean()),
        "valid_overlap_rate_audit_only": float(overlap[valid & r_rows & complete].mean()),
        "features": int(features.shape[1]),
    }), flush=True)

    seed_predictions = []
    for seed_index in range(args.n_seeds):
        model = CatBoostClassifier(**parameters(args, 101 * seed_index))
        model.fit(
            features.loc[train], overlap[train],
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        seed_predictions.append(model.predict_proba(features.loc[valid])[:, 1])
        print(f"completed seed={seed_index + 1}/{args.n_seeds}", flush=True)
    overlap_probability = np.mean(seed_predictions, axis=0)

    with np.load(
        ROOT / f"research/v34_categorical_failure_lowcard_no_ids_hl2_{args.valid_year}.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
        fold_target = archive["target"].astype(float)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        v24 = np.clip(archive["blended"][active].astype(float), .005, .995)
    with np.load(
        ROOT / f"research/v35_lowcard_direct_hl2_s3_{args.valid_year}.npz",
        allow_pickle=True,
    ) as archive:
        direct = np.clip(archive["prediction"].astype(float), .005, .995)
    first = sigmoid(.825 * logit(v24) + .175 * logit(failure))
    base = sigmoid(.90 * logit(first) + .10 * logit(direct))
    if args.valid_year == 2024:
        with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
            replay = np.clip(
                archive["blended"][archive["season"] == 2024].astype(float),
                .005, .995,
            )
        if not np.allclose(base, replay, atol=2e-7):
            raise ValueError("reconstructed v38 does not match the frozen OOF")
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("prediction rows do not align")

    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    reports = []
    for inclusion_scale in np.round(np.arange(0., 1.501, .05), 3):
        corrected = np.clip(
            failure + inclusion_scale * overlap_probability, .005, .995,
        )
        direction = logit(corrected) - logit(base)
        for weight in np.round(np.arange(-.10, .301, .025), 3):
            candidate = base.copy()
            candidate[valid_r] = sigmoid(
                logit(base[valid_r]) + weight * direction[valid_r]
            )
            result = score(fold_target, candidate, blocks, game_type)
            gains = {name: result[name] - baseline[name] for name in result}
            reports.append({
                "inclusion_scale": float(inclusion_scale),
                "weight": float(weight), "scores": result, "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"], row["gains"]["R"]),
        row["scores"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "baseline": baseline,
        "overlap_prediction_mean_R": float(overlap_probability[valid_r].mean()),
        "overlap_target_rate_R_audit_only": float(overlap[valid & r_rows & complete].mean()),
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:50],
        "best_score": sorted(
            reports, key=lambda row: row["scores"]["all"], reverse=True,
        )[:50],
    }
    output = ROOT / "research" / f"v51_recent_r_overlap_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        overlap_probability=overlap_probability.astype(np.float32),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
