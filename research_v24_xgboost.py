"""Chronological GPU XGBoost screen over the v24 prediction.

The goal is model-family diversity, not another CatBoost parameter tweak.  All
features are row-local and time-safe; Trackman summaries use only seasons before
the row season.  Candidate predictions are evaluated as small logit-space
increments over v24 on both all rows and the R/F regimes separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_combined_candidate import logit, sigmoid
from trackman_context import attach_context, pitcher_mapping, prepare_trackman


ID_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "team_matchup",
)
VARIANTS = {
    "noid_d6": dict(
        drop_ids=True, max_depth=6, min_child_weight=100., reg_lambda=100.,
        n_estimators=1200, learning_rate=.025, max_bin=128,
    ),
    "noid_d8": dict(
        drop_ids=True, max_depth=8, min_child_weight=250., reg_lambda=200.,
        n_estimators=1400, learning_rate=.02, max_bin=128,
    ),
    "noid_leaves64": dict(
        drop_ids=True, max_depth=0, max_leaves=64, min_child_weight=150.,
        reg_lambda=150., n_estimators=1400, learning_rate=.02, max_bin=128,
        grow_policy="lossguide",
    ),
    "numeric_id_d6": dict(
        drop_ids=False, max_depth=6, min_child_weight=100., reg_lambda=100.,
        n_estimators=1200, learning_rate=.025, max_bin=128,
    ),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def masks(rows: pd.DataFrame):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def evaluate(target, base, prediction, rows):
    segment = masks(rows)
    direction = logit(prediction) - logit(base)
    reports = []
    for gate in ("all", "regular", "futures"):
        gated = direction * segment[gate]
        for weight in np.arange(-.10, .401, .01):
            candidate = sigmoid(logit(base) + weight * gated)
            gains = {
                name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
                for name, mask in segment.items() if mask.any()
            }
            reports.append({
                "gate": gate, "weight": float(weight), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
                "min_month": min(
                    gains[name]
                    for name in ("months_3_5", "months_6_7", "months_8_11")
                ),
            })
    robust = max(
        reports,
        key=lambda item: (
            item["min_half"], item["min_quarter"], item["gains"]["all"],
        ),
    )
    best_all = max(reports, key=lambda item: item["gains"]["all"])
    return {"robust_best": robust, "best_all": best_all}


def main():
    args = arguments()
    selected = args.variants.split(",")
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    context = attach_context(data, prepare_trackman(trackman, mapping))
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1).drop(columns=["game_month"])
    # XGBoost treats these compact codes as ordinary numeric variables.  The
    # no-ID variants deliberately remove arbitrary high-cardinality ordering.
    features = features.replace([np.inf, -np.inf], np.nan)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    rows = data.loc[valid].reset_index(drop=True)
    with np.load(root / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        y = archive["target"][active].astype(float)
        base = archive["blended"][active].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v24 OOF rows do not align")

    predictions = {}
    reports = {}
    for offset, name in enumerate(selected):
        configuration = VARIANTS[name].copy()
        drop_ids = configuration.pop("drop_ids")
        columns = [
            column for column in features
            if not (drop_ids and column in ID_COLUMNS)
        ]
        model = XGBClassifier(
            **configuration, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cuda", sampling_method="gradient_based",
            subsample=.85, colsample_bytree=.85, reg_alpha=.05,
            random_state=args.seed + offset, verbosity=1, n_jobs=-1,
        )
        model.fit(features.loc[train, columns], target[train])
        prediction = model.predict_proba(features.loc[valid, columns])[:, 1]
        report = evaluate(y, base, prediction, rows)
        report.update({
            "standalone_bss": bss(y, prediction),
            "standalone_regular_bss": bss(
                y[rows["game_type"].eq("R")],
                prediction[rows["game_type"].eq("R")],
            ),
            "standalone_futures_bss": bss(
                y[rows["game_type"].eq("F")],
                prediction[rows["game_type"].eq("F")],
            ),
            "prediction_mean": float(prediction.mean()),
            "feature_count": len(columns), "configuration": configuration,
        })
        predictions[name] = prediction.astype(np.float32)
        reports[name] = report
        print(json.dumps({"variant": name, **report}), flush=True)

    output = root / "research" / f"v24_xgboost_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        names=np.asarray(selected), reports_json=np.asarray(json.dumps(reports)),
        **predictions,
    )
    ranking = sorted(
        reports.items(),
        key=lambda item: (
            item[1]["robust_best"]["min_half"],
            item[1]["robust_best"]["min_quarter"],
            item[1]["robust_best"]["gains"]["all"],
        ), reverse=True,
    )
    print(json.dumps({"ranking": ranking}, indent=2), flush=True)
    print(f"Saved {output}; mapped_pitchers={len(mapping)}; "
          f"minimum_mapping_confidence={mapping_report['confidence'].min():.6f}", flush=True)


if __name__ == "__main__":
    main()
