"""Screen row-local confidence recovered from recent-game rate fractions."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from recent_window_features import recent_window_features
from research_inferred_pitch_priors import bss
from research_v35_lowcard_direct_cat import parameters
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1100)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def score(target, prediction, blocks, game_type):
    result = {
        name: float(bss(target[active], prediction[active]))
        for name, active in blocks.items()
    }
    for regime in ("R", "F"):
        active = game_type == regime
        result[regime] = float(bss(target[active], prediction[active]))
    return result


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    recent = recent_window_features(raw)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    features = pd.concat([features, recent], axis=1)
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
        "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "features": int(features.shape[1]), "new_features": int(recent.shape[1]),
        "fraction_observed": {
            str(window): float(
                recent.loc[valid, f"recent{window}_fraction_observed"].mean()
            ) for window in (1, 3, 5)
        },
        "fraction_n_monotone": float(
            recent.loc[valid, "recent_fraction_n_monotone"].mean()
        ),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }), flush=True)

    members = []
    for seed_index in range(args.n_seeds):
        print(f"v58 seed={seed_index + 1}/{args.n_seeds}", flush=True)
        model = CatBoostClassifier(**parameters(args, seed_index))
        model.fit(
            features.loc[train], target[train], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        members.append(model.predict_proba(features.loc[valid])[:, 1])
    prediction = np.mean(members, axis=0)

    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        fold_target = archive["target"][active].astype(float)
        base = np.clip(archive["blended"][active].astype(float), .005, .995)
    if not np.allclose(fold_target, target[valid]):
        raise ValueError("v54 and train.csv rows do not align")
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    blocks = masks(len(base))
    baseline = score(fold_target, base, blocks, game_type)
    direction = logit(np.clip(prediction, .005, .995)) - logit(base)
    reports = []
    for gate in ("all", "R", "F"):
        selected = np.ones(len(base), bool) if gate == "all" else game_type == gate
        for weight in np.round(np.arange(-.10, .401, .0125), 4):
            candidate = base.copy()
            candidate[selected] = sigmoid(
                logit(base[selected]) + weight * direction[selected]
            )
            scores = score(fold_target, candidate, blocks, game_type)
            gains = {name: scores[name] - baseline[name] for name in scores}
            reports.append({
                "gate": gate, "weight": float(weight),
                "scores": scores, "gains": gains,
                "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                "min_half": float(min(gains["h1"], gains["h2"])),
            })
    robust_key = lambda row: (
        min(row["min_quarter"], row["min_half"],
            row["gains"]["R"], row["gains"]["F"]),
        row["gains"]["all"],
    )
    diagnostics = {
        "valid_year": args.valid_year, "baseline": baseline,
        "standalone": score(fold_target, prediction, blocks, game_type),
        "correlation_base": float(np.corrcoef(prediction, base)[0, 1]),
        "best_robust": sorted(reports, key=robust_key, reverse=True)[:40],
        "best_score": sorted(
            reports, key=lambda row: row["gains"]["all"], reverse=True,
        )[:40],
        "new_feature_columns": list(recent.columns),
        "row_independent": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    output = ROOT / "research" / (
        f"v58_recent_fraction_hl{args.half_life:g}_s{args.n_seeds}_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output, target=fold_target.astype(np.float32),
        base=base.astype(np.float32), prediction=prediction.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
