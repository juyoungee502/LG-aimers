"""Screen batter-exposure feature ablations in the weighted CatBoost component."""
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
from train_weighted_component import MODEL_CAT_COLUMNS, parameters, weights


SEEDS = (42, 43, 44)
DROP_COLUMNS = {
    "raw": ("asof_batter_n", "log1p_asof_batter_n"),
    "exposure": (
        "asof_batter_n", "batter_career_success_count", "batter_season_n",
        "batter_season_success_count", "batter_season_log_n",
        "batter_season_weight_s10", "batter_season_weight_s25",
        "batter_season_weight_s50", "batter_season_weight_s100",
        "batter_season_weight_s200", "log1p_asof_batter_n",
    ),
}


def segment_gains(target, base, candidate, rows):
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }
    return {
        name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
        for name, mask in masks.items() if mask.any()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in MODEL_CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = raw["season"].to_numpy(np.int16)
    train_mask = seasons < args.valid_year
    valid_mask = seasons == args.valid_year

    predictions = {}
    for variant, drop_columns in DROP_COLUMNS.items():
        matrix = features.drop(columns=list(drop_columns))
        seed_predictions = []
        for seed in SEEDS:
            model = CatBoostClassifier(**parameters(args, seed))
            model.fit(
                matrix.loc[train_mask], target[train_mask],
                sample_weight=weights(seasons[train_mask], args.valid_year - 1),
            )
            seed_predictions.append(model.predict_proba(matrix.loc[valid_mask])[:, 1])
            print(f"{variant} complete: year={args.valid_year}, seed={seed}", flush=True)
        predictions[variant] = np.mean(seed_predictions, axis=0)

    with np.load(root / "outputs" / "v23_oof_predictions.npz") as loaded:
        v23 = {key: loaded[key] for key in loaded.files}
    fold = v23["season"] == args.valid_year
    y = v23["target"][fold].astype(float)
    base = v23["blended"][fold].astype(float)
    names = v23["model_names"].astype(str).tolist()
    old_component = v23["predictions"][fold, names.index("weighted_catboost")].astype(float)
    if not np.allclose(y, target[valid_mask]):
        raise ValueError("v23 OOF rows do not align")
    rows = raw.loc[valid_mask].reset_index(drop=True)
    other = rows["strikes_before"].ne(2).to_numpy()
    # Exact coefficient of this component in the v15 segment blend. Later v17-v23
    # stages are close to identity, so the paired delta remains the cleanest axis.
    segment_coefficient = 1.038038555780343 * 0.42775971722734324
    reports = []
    for variant, prediction in predictions.items():
        direction = np.zeros(len(y), dtype=float)
        direction[other] = segment_coefficient * (
            prediction[other] - old_component[other]
        )
        for scale in np.arange(-.5, 1.501, .05):
            candidate = np.clip(base + scale * direction, .005, .995)
            gains = segment_gains(y, base, candidate, rows)
            reports.append({
                "variant": variant, "scale": float(scale), "gains": gains,
                "min_half": min(gains["first_half"], gains["second_half"]),
                "min_month_segment": min(
                    gains[name] for name in (
                        "months_3_5", "months_6_7", "months_8_11",
                    )
                ),
                "standalone_bss": bss(y, prediction),
            })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    output = root / "research" / f"v23_batter_exposure_ablation_{args.valid_year}.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        old_component=old_component.astype(np.float32),
        raw=predictions["raw"].astype(np.float32),
        exposure=predictions["exposure"].astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    exact = [row for row in reports if abs(row["scale"] - 1.0) < 1e-8]
    print(json.dumps({
        "valid_year": args.valid_year, "exact_replacements": exact,
        "top": reports[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
