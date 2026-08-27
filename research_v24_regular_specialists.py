"""Screen regular-season-only CatBoost specialists against the v24 OOF.

This is a research script, not a submission trainer.  Every validation row is
predicted from a model fitted only on earlier seasons, Trackman context is built
only from earlier seasons, and the candidate is evaluated as a paired
replacement of v24's existing no-month command specialist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_combined_candidate import logit, sigmoid
from trackman_context import attach_context, pitcher_mapping, prepare_trackman
from train_trackman_context_specialist import BLEND_WEIGHT
from v24_robust_candidate import POLICY


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup",
    "team_matchup",
]

# The first screen deliberately changes one modelling decision at a time.
VARIANTS = {
    "r_uniform_d8": {
        "route": "regular", "weight": None, "feature_set": "no_month",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_hl3_d8": {
        "route": "regular", "weight": 3., "feature_set": "no_month",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_hl1p5_d8": {
        "route": "regular", "weight": 1.5, "feature_set": "no_month",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_recent3_d8": {
        "route": "regular_recent3", "weight": None,
        "feature_set": "no_month", "depth": 8, "iterations": 1600,
        "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_uniform_d6": {
        "route": "regular", "weight": None, "feature_set": "no_month",
        "depth": 6, "iterations": 1200, "learning_rate": .02,
        "l2_leaf_reg": 100., "native_cat": False,
    },
    "r_hl3_d6": {
        "route": "regular", "weight": 3., "feature_set": "no_month",
        "depth": 6, "iterations": 1200, "learning_rate": .02,
        "l2_leaf_reg": 100., "native_cat": False,
    },
    "r_uniform_d8_timeless": {
        "route": "regular", "weight": None, "feature_set": "timeless",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_hl3_d8_timeless": {
        "route": "regular", "weight": 3., "feature_set": "timeless",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": False,
    },
    "r_native_d6": {
        "route": "regular", "weight": 3., "feature_set": "no_month",
        "depth": 6, "iterations": 1200, "learning_rate": .02,
        "l2_leaf_reg": 100., "native_cat": True,
    },
    "r_native_d8": {
        "route": "regular", "weight": 3., "feature_set": "no_month",
        "depth": 8, "iterations": 1600, "learning_rate": .01631820635235777,
        "l2_leaf_reg": 509.6419153575998, "native_cat": True,
    },
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated variant names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    return parser.parse_args()


def recency_weights(seasons: np.ndarray, reference: int, half_life: float | None):
    if half_life is None:
        return None
    age = np.maximum(0., reference - seasons.astype(np.float64))
    return np.exp(-np.log(2.) * age / half_life).astype(np.float32)


def parameters(configuration, args):
    result = dict(
        iterations=configuration["iterations"],
        learning_rate=configuration["learning_rate"],
        depth=configuration["depth"], l2_leaf_reg=configuration["l2_leaf_reg"],
        random_strength=(1. if configuration["depth"] == 6
                         else 2.9151912613602535),
        border_count=32, bootstrap_type="Bayesian",
        bagging_temperature=(.5 if configuration["depth"] == 6
                             else .36881602504480515),
        loss_function="Logloss", eval_metric="Logloss",
        task_type=args.task_type, random_seed=args.seed,
        allow_writing_files=False, verbose=0,
    )
    if configuration["native_cat"]:
        result.update(max_ctr_complexity=1, one_hot_max_size=32)
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def segment_masks(rows: pd.DataFrame):
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


def evaluate(target, base, current, candidate, rows):
    regular = rows["game_type"].eq("R").to_numpy()
    # v24 applies 1.70 times the 0.35 component replacement.  This direction
    # therefore means exactly replacing the deployed no-month member.
    direction = np.zeros(len(target), dtype=np.float64)
    direction[regular] = POLICY["command_no_month"] * BLEND_WEIGHT * (
        logit(candidate[regular]) - logit(current[regular])
    )
    masks = segment_masks(rows)
    curve = []
    for scale in np.arange(-.50, 1.501, .05):
        prediction = sigmoid(logit(base) + scale * direction)
        gains = {
            name: bss(target[mask], prediction[mask]) - bss(target[mask], base[mask])
            for name, mask in masks.items() if mask.any()
        }
        curve.append({
            "scale": float(scale), "gains": gains,
            "min_half": min(gains["first_half"], gains["second_half"]),
            "min_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
            "min_month": min(
                gains[name] for name in ("months_3_5", "months_6_7", "months_8_11")
            ),
        })
    exact = min(curve, key=lambda item: abs(item["scale"] - 1.))
    robust = max(
        curve,
        key=lambda item: (
            item["min_half"], item["min_quarter"], item["gains"]["all"],
        ),
    )
    return direction, {
        "current_regular_bss": bss(target[regular], current[regular]),
        "candidate_regular_bss": bss(target[regular], candidate[regular]),
        "exact_replacement": exact, "robust_best": robust,
    }


def main():
    args = arguments()
    selected = args.variants.split(",")
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
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
    base_features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(base_features, data)
    base_features = add_state_interactions(base_features)
    base_features = pd.concat([base_features, context], axis=1).drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        base_features[column] = base_features[column].fillna(-1).astype(np.int32)

    seasons = data["season"].to_numpy(np.int16)
    valid = seasons == args.valid_year
    rows = data.loc[valid].reset_index(drop=True)
    with np.load(root / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        y = archive["target"][active].astype(float)
        v24 = archive["blended"][active].astype(float)
    with np.load(root / f"research/v23_trackman_no_month_{args.valid_year}.npz") as archive:
        current = archive["no_month_specialist"].astype(float)
    if not np.allclose(y, target[valid]):
        raise ValueError("v24 OOF rows do not align")

    predictions = {}
    reports = {}
    for name in selected:
        configuration = VARIANTS[name]
        train = (seasons < args.valid_year) & data["game_type"].eq("R").to_numpy()
        if configuration["route"] == "regular_recent3":
            train &= seasons >= args.valid_year - 3
        features = base_features
        if configuration["feature_set"] == "timeless":
            features = features.drop(columns=["season"])
        model = CatBoostClassifier(**parameters(configuration, args))
        model.fit(
            features.loc[train], target[train],
            sample_weight=recency_weights(
                seasons[train], args.valid_year - 1, configuration["weight"],
            ),
            cat_features=(CAT_COLUMNS if configuration["native_cat"] else None),
        )
        prediction = model.predict_proba(features.loc[valid])[:, 1]
        direction, report = evaluate(y, v24, current, prediction, rows)
        report["configuration"] = configuration
        report["train_rows"] = int(train.sum())
        reports[name] = report
        predictions[name] = prediction.astype(np.float32)
        predictions[f"{name}_direction"] = direction.astype(np.float32)
        print(json.dumps({"variant": name, **report}), flush=True)

    output = root / "research" / f"v24_regular_specialists_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=v24.astype(np.float32),
        current=current.astype(np.float32), names=np.asarray(selected),
        reports_json=np.asarray(json.dumps(reports)), **predictions,
    )
    summary = sorted(
        reports.items(), key=lambda item: (
            item[1]["exact_replacement"]["gains"]["all"],
            item[1]["exact_replacement"]["min_half"],
        ), reverse=True,
    )
    print(json.dumps({
        "valid_year": args.valid_year, "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "ranking": [
            {
                "variant": name,
                "standalone_delta": report["candidate_regular_bss"]
                                    - report["current_regular_bss"],
                "exact_gain": report["exact_replacement"]["gains"]["all"],
                "exact_min_half": report["exact_replacement"]["min_half"],
                "exact_min_quarter": report["exact_replacement"]["min_quarter"],
                "robust_best": report["robust_best"],
            }
            for name, report in summary
        ],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
