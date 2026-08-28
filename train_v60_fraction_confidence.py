"""Train the six-pair roster-safe recent-fraction correction over v54."""
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
from research_v40_failure_seed_stability import masks
from research_v58_fraction_roster_stability import (
    pitcher_season_exposure, roster_masks,
)


ROOT = Path(__file__).resolve().parent
LOW_CARD_CATEGORIES = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
)
MODEL_COUNT = 6
CORRECTION_WEIGHT = .75
RECENT1_MIN_N = 30.
MIN_SEASON_EXPOSURE = 100.
HALF_LIFE = 2.
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--correction-weight", type=float, default=CORRECTION_WEIGHT)
    parser.add_argument(
        "--output-version",
        choices=("v60_fraction_confidence", "v62_fraction_full"),
        default="v60_fraction_confidence",
    )
    return parser.parse_args()


def parameters(args, seed_index):
    result = dict(
        iterations=1400, learning_rate=.018, depth=7,
        l2_leaf_reg=360., random_strength=2.8, bagging_temperature=1.2,
        border_count=32, bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", random_seed=5900 + 101 * seed_index,
        task_type=args.task_type, thread_count=args.threads,
        allow_writing_files=False, verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, gpu_ram_part=.90)
    return result


def score(target, prediction, active):
    return float(bss(target[active], prediction[active]))


def audited_oof(raw, correction_weight, output_version):
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54_archive = {key: archive[key] for key in archive.files}
    year = 2024
    active = v54_archive["season"] == year
    target = v54_archive["target"][active].astype(float)
    base = np.clip(v54_archive["blended"][active].astype(float), .005, .995)
    references = []
    predictions = []
    valid_f = None
    for suffix in ("", "_o3"):
        with np.load(
            ROOT / "research" / f"v59_f_fraction_s3{suffix}_{year}.npz"
        ) as archive:
            if valid_f is None:
                valid_f = archive["valid_f"].astype(bool)
            elif not np.array_equal(valid_f, archive["valid_f"].astype(bool)):
                raise ValueError("independent v59 groups do not align")
            references.append(archive["reference"].astype(float))
            predictions.append(archive["prediction"].astype(float))
    rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    recent = recent_window_features(rows)
    selected = (
        valid_f
        & (recent["recent1_reduced_n"].to_numpy(float) >= RECENT1_MIN_N)
        & (pitcher_season_exposure(raw, rows, year) > MIN_SEASON_EXPOSURE)
    )
    direction = np.zeros(len(rows), dtype=float)
    direction[valid_f] = (
        np.mean(predictions, axis=0) - np.mean(references, axis=0)
    )
    candidate = np.clip(
        base + correction_weight * direction * selected, .005, .995,
    )
    segments = {
        **masks(len(rows)),
        "R": ~valid_f, "F": valid_f,
        **roster_masks(raw, rows, year),
    }
    baseline = {name: score(target, base, mask) for name, mask in segments.items()}
    scores = {name: score(target, candidate, mask) for name, mask in segments.items()}
    gains = {name: scores[name] - baseline[name] for name in segments}

    audit_key = f"weight_{correction_weight:.2f}"
    audit = json.loads(
        (ROOT / "research/v59_group_stability.json").read_text(encoding="utf-8")
    )["summary"][audit_key]
    if (
        audit["minimum_all_gain"] < 2.9
        or audit["minimum_quarter_gain"] <= 0.
        or audit["minimum_affected_roster_gain"] <= 0.
        or audit["minimum_bootstrap_p05"] <= 0.
        or gains["all"] < 3.0
        or min(gains[f"q{i}"] for i in range(1, 5)) <= 0.
    ):
        raise RuntimeError(
            f"fraction promotion gate failed: independent_groups={audit}, gains={gains}"
        )
    short_version = "v62" if output_version == "v62_fraction_full" else "v60"
    upgraded = v54_archive["blended"].astype(np.float64).copy()
    upgraded[active] = candidate
    np.savez_compressed(
        ROOT / "outputs" / f"{short_version}_oof_predictions.npz",
        **{key: value for key, value in v54_archive.items() if key != "blended"},
        blended=upgraded,
    )
    return {
        "baseline": baseline, "scores": scores, "gains": gains,
        "selected_rows": int(selected.sum()),
        "correction_weight": correction_weight,
        "output_version": output_version,
        "independent_group_audit": audit,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }


def main():
    args = arguments()
    model_dir = ROOT / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v54_roster_robust_command", "v60_fraction_confidence",
        "v62_fraction_full",
    ):
        raise ValueError(f"Expected v54/v60 artifacts, got {metadata.get('version')}")

    full = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = full[TARGET_COL].astype(np.float32)
    target = target_series.to_numpy(np.float32)
    raw = full.drop(columns=[TARGET_COL])
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
    fraction_features = pd.concat([base_features, recent], axis=1)
    seasons = raw["season"].to_numpy(np.int16)
    futures = raw["game_type"].astype(str).eq("F").to_numpy()
    age = int(seasons.max()) - seasons[futures].astype(float)
    sample_weight = np.exp(
        -np.log(2.) * age / HALF_LIFE
    ).astype(np.float32)

    model_dir.mkdir(parents=True, exist_ok=True)
    for seed_index in range(MODEL_COUNT):
        print(f"Training v60 paired base {seed_index + 1}/{MODEL_COUNT}", flush=True)
        model = CatBoostClassifier(**parameters(args, seed_index))
        model.fit(
            base_features.loc[futures], target[futures], sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        model.save_model(str(model_dir / f"catboost_v60_base_{seed_index}.cbm"))
        print(f"Training v60 paired fraction {seed_index + 1}/{MODEL_COUNT}", flush=True)
        model = CatBoostClassifier(**parameters(args, seed_index))
        model.fit(
            fraction_features.loc[futures], target[futures],
            sample_weight=sample_weight,
            cat_features=list(LOW_CARD_CATEGORIES),
        )
        model.save_model(str(model_dir / f"catboost_v60_fraction_{seed_index}.cbm"))

    report = audited_oof(raw, args.correction_weight, args.output_version)
    names = list(metadata.get("model_names", []))
    if "v60_fraction_confidence" not in names:
        names.append("v60_fraction_confidence")
    metadata["model_names"] = names
    metadata["version"] = args.output_version
    metadata["v60_fraction_confidence"] = {
        "base_feature_columns": list(base_features.columns),
        "fraction_feature_columns": list(fraction_features.columns),
        "categorical_columns": list(LOW_CARD_CATEGORIES),
        "model_count": MODEL_COUNT,
        "correction_weight": args.correction_weight,
        "recent1_min_reduced_n": RECENT1_MIN_N,
        "minimum_pitcher_season_exposure": MIN_SEASON_EXPOSURE,
        "training_game_type": "F",
        "raw_player_ids_used": False,
        "row_independent_inference": True,
        "current_pitch_type_used": False,
        "forbidden_2025_trackman_used": False,
    }
    short_version = "v62" if args.output_version == "v62_fraction_full" else "v60"
    metadata.setdefault("training_info", {})[f"{short_version}_validation"] = report
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(f"{short_version} validation: {json.dumps(report)}", flush=True)
    print(
        f"Stored {short_version} paired models, metadata, and OOF diagnostics",
        flush=True,
    )


if __name__ == "__main__":
    main()
