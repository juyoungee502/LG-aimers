"""Ablate recent-fraction feature groups against frozen paired base members."""
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
from research_v59_f_fraction_specialist import LOW_CARD_CATEGORIES, parameters


ROOT = Path(__file__).resolve().parent
VARIANTS = ("confidence", "counts", "shrinkage", "core", "window1")
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--seed-offset", type=int, default=0, choices=(0, 3))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def variant_columns(recent, variant):
    columns = list(recent.columns)
    confidence = [
        column for column in columns
        if (
            "reduced_n" in column
            or "fraction_observed" in column
            or "_weight_s" in column
            or "n_monotone" in column
            or "n_ratio" in column
            or column.endswith("_valid")
        )
    ]
    counts = [
        column for column in columns
        if column.endswith("_success_count") or column.endswith("_middle_count")
    ]
    shrinkage = [column for column in columns if "_success_s" in column]
    if variant == "confidence":
        selected = confidence
    elif variant == "counts":
        selected = confidence + counts
    elif variant == "shrinkage":
        selected = confidence + shrinkage
    elif variant == "core":
        selected = confidence + counts + shrinkage
    elif variant == "window1":
        selected = [
            column for column in columns
            if column.startswith("recent1_") or column == "recent_fraction_n_monotone"
        ]
    else:
        raise ValueError(variant)
    return list(dict.fromkeys(selected))


def main():
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    recent = recent_window_features(raw)
    bases = training_history_arrays(raw, target_series)
    base_features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(base_features, raw)
    base_features = add_state_interactions(base_features)
    base_features = base_features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in base_features
    ])
    for column in LOW_CARD_CATEGORIES:
        base_features[column] = base_features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    futures = raw["game_type"].astype(str).eq("F").to_numpy()
    train = (seasons < args.valid_year) & futures
    valid = (seasons == args.valid_year) & futures
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / args.half_life
    ).astype(np.float32)
    suffix = "" if args.seed_offset == 0 else f"_o{args.seed_offset}"
    with np.load(
        ROOT / "research" / f"v59_f_fraction_s3{suffix}_{args.valid_year}.npz"
    ) as archive:
        fold_target = archive["target"].astype(float)
        v54 = archive["base"].astype(float)
        valid_f = archive["valid_f"].astype(bool)
        references = archive["reference_members"].astype(float)
    if references.shape[0] != args.n_seeds or int(valid_f.sum()) != int(valid.sum()):
        raise ValueError("frozen reference members do not align")

    for variant in args.variants:
        selected_columns = variant_columns(recent, variant)
        features = pd.concat([base_features, recent[selected_columns]], axis=1)
        members = []
        for local_seed in range(args.n_seeds):
            paired_seed = args.seed_offset + local_seed
            print(
                f"v61 {variant}: seed={paired_seed + 1}, year={args.valid_year}, "
                f"new_features={len(selected_columns)}", flush=True,
            )
            model = CatBoostClassifier(**parameters(args, paired_seed))
            model.fit(
                features.loc[train], target[train], sample_weight=sample_weight,
                cat_features=list(LOW_CARD_CATEGORIES),
            )
            members.append(model.predict_proba(features.loc[valid])[:, 1])
        prediction = np.mean(members, axis=0)
        reference = references.mean(axis=0)
        diagnostics = {
            "variant": variant, "valid_year": args.valid_year,
            "seed_offset": args.seed_offset,
            "new_feature_count": len(selected_columns),
            "new_feature_columns": selected_columns,
            "reference_bss_F": float(bss(target[valid], reference)),
            "variant_bss_F": float(bss(target[valid], prediction)),
            "standalone_gain_F": float(
                bss(target[valid], prediction) - bss(target[valid], reference)
            ),
            "row_independent": True,
            "current_pitch_type_used": False,
            "forbidden_2025_trackman_used": False,
        }
        output = ROOT / "research" / (
            f"v61_fraction_{variant}_s{args.n_seeds}{suffix}_{args.valid_year}.npz"
        )
        np.savez_compressed(
            output, target=fold_target.astype(np.float32),
            base=v54.astype(np.float32), valid_f=valid_f,
            reference=reference.astype(np.float32),
            prediction=prediction.astype(np.float32),
            prediction_members=np.asarray(members, dtype=np.float32),
            diagnostics_json=np.asarray(json.dumps(diagnostics)),
        )
        print(json.dumps(diagnostics, indent=2), flush=True)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
