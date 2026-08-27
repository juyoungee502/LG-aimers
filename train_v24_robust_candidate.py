"""Train and package the time-safe v24 command/resolution candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_combined_candidate import add_pitcher_season_exposure, logit, sigmoid
from research_v23_context_deviation import deviation, masks
from trackman_context import FEATURE_COLUMNS, attach_context, pitcher_mapping, prepare_trackman
from v24_robust_candidate import (
    COMMAND_BLEND_SCALE, POLICY, RESOLUTION_CONTEXTS,
    freeze_command, freeze_pitcher_season_origins, freeze_pressure,
    freeze_resolution_center, resolution_context_frame, resolution_label,
    time_safe_command_features,
)


COMMAND_SEEDS = (42, 2025, 3407)
RESOLUTION_SEEDS = (4501, 4502, 4503)
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def command_parameters(args, seed):
    result = dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", task_type=args.task_type,
        random_seed=seed, allow_writing_files=False, verbose=0,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def resolution_parameters(args, seed):
    result = dict(
        iterations=1000, depth=6, learning_rate=.025,
        loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=150.,
        random_strength=1., border_count=32, bootstrap_type="Bayesian",
        bagging_temperature=.5, task_type=args.task_type,
        random_seed=seed, allow_writing_files=False, verbose=0,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def fit_command_models(features, target, model_dir, label, args):
    for index, seed in enumerate(COMMAND_SEEDS):
        model = CatBoostClassifier(**command_parameters(args, seed))
        model.fit(features, target)
        model.save_model(str(model_dir / f"catboost_v24_command_{label}_{index}.cbm"))
        print(f"v24 command model complete: label={label} seed={seed}", flush=True)


def fit_resolution_models(features, label, context, mode, model_dir, args):
    members = []
    mode_offset = list(RESOLUTION_CONTEXTS).index(mode)
    for index, seed in enumerate(RESOLUTION_SEEDS):
        model = CatBoostRegressor(
            **resolution_parameters(args, seed + 100 * mode_offset),
        )
        model.fit(features, label)
        members.append(model.predict(features))
        model.save_model(str(
            model_dir / f"catboost_v24_resolution_{mode}_{index}.cbm"
        ))
        print(f"v24 resolution model complete: mode={mode} seed={seed}", flush=True)
    prediction = np.mean(members, axis=0)
    return freeze_resolution_center(context, prediction, mode)


def audited_oof(root: Path, data: pd.DataFrame):
    with np.load(root / "outputs/v23_oof_predictions.npz") as archive:
        v23 = {key: archive[key] for key in archive.files}
    exposure = add_pitcher_season_exposure(data[[
        "season", "pitcher_id", "asof_pitcher_n",
    ]])
    upgraded = v23["blended"].astype(np.float64).copy()
    reports = {}
    for year in (2023, 2024):
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        active = v23["season"] == year
        target = v23["target"][active].astype(float)
        base = v23["blended"][active].astype(float)
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"v23 rows do not align for {year}")
        with np.load(root / f"research/v23_trackman_no_month_{year}.npz") as z:
            no_month = z["direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}.npz") as z:
            full = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}_w1.npz") as z:
            recent = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_conditional_resolution_{year}.npz") as z:
            resolution_names = z["names"].astype(str).tolist()
            resolution_values = z["directions"].astype(float)
        expected_names = list(RESOLUTION_CONTEXTS)
        resolution_values = resolution_values[:, [
            resolution_names.index(name) for name in expected_names
        ]]
        season_exposure = exposure.loc[
            exposure["season"].eq(year), "pitcher_season_n",
        ].to_numpy(float)
        regular = rows["game_type"].eq("R").to_numpy(float)
        futures = rows["game_type"].eq("F").to_numpy(float)
        early = (season_exposure <= 600.).astype(float)
        command = (
            POLICY["global_logit_shift"]
            + regular * (
                POLICY["command_no_month"] * no_month
                + early * POLICY["command_full"] * full
                + early * POLICY["command_recent"] * recent
            )
        )
        prediction = sigmoid(logit(base) + command)
        resolution_weights = np.asarray([
            POLICY["f_count"], POLICY["f_hands"], POLICY["f_runners"],
        ])
        source = data.loc[data["season"].lt(year)].copy()
        source["pressure_state"] = np.where(
            (source["balls_before"].eq(3) & source["strikes_before"].eq(2)), 2,
            np.where(
                source["balls_before"].eq(3) | source["strikes_before"].eq(2),
                1, 0,
            ),
        ).astype(np.int8)
        source["season_relative_target"] = (
            source["control_success"]
            - source.groupby(["season", "game_type"], observed=True)[
                "control_success"
            ].transform("mean")
        )
        query = rows.copy()
        query["pressure_state"] = np.where(
            (query["balls_before"].eq(3) & query["strikes_before"].eq(2)), 2,
            np.where(
                query["balls_before"].eq(3) | query["strikes_before"].eq(2),
                1, 0,
            ),
        ).astype(np.int8)
        pressure = deviation(
            source, query,
            ("pitcher_id", "pressure_state", "batter_hand"),
            1200., "season_relative_target",
        )
        prediction = np.clip(
            prediction
            + futures * (resolution_values @ resolution_weights)
            + POLICY["pressure_hand"] * pressure,
            .005, .995,
        )
        segment_masks = masks(rows)
        gains = {
            name: bss(target[mask], prediction[mask]) - bss(target[mask], base[mask])
            for name, mask in segment_masks.items() if mask.any()
        }
        reports[str(year)] = {
            "v23_bss": bss(target, base), "v24_bss": bss(target, prediction),
            "gain": gains["all"], "segment_gains": gains,
        }
        upgraded[active] = prediction
    if (
        reports["2023"]["gain"] < 35.
        or reports["2024"]["gain"] < 17.
        or min(reports["2024"]["segment_gains"].values()) <= 0.
    ):
        raise RuntimeError(f"v24 promotion gate failed: {reports}")
    np.savez_compressed(
        root / "outputs/v24_oof_predictions.npz",
        **{key: value for key, value in v23.items() if key != "blended"},
        blended=upgraded,
    )
    return reports


def main():
    args = arguments()
    root = Path(__file__).resolve().parent
    model_dir = root / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v23_probability_residual_portfolio", "v24_robust_command_resolution",
    ):
        raise ValueError(f"Expected v23/v24 artifacts, got {metadata.get('version')}")

    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(float)
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    trackman_features = attach_context(data, trackman)

    bases = training_history_arrays(data, target_series)
    base_features = engineer_features(
        data, *bases, global_prior=float(target.mean()),
    )
    add_training_component_features(base_features, data)
    base_features = add_state_interactions(base_features)
    for column in metadata["cat_features"]:
        base_features[column] = base_features[column].fillna(-1).astype(np.int32)
    if list(base_features.columns) != metadata["feature_columns"]:
        raise ValueError("v24 base feature schema differs from v23 metadata")

    no_month_features = pd.concat([
        base_features.reset_index(drop=True), trackman_features.reset_index(drop=True),
    ], axis=1).drop(columns=["game_month"])
    for column in CAT_COLUMNS:
        no_month_features[column] = no_month_features[column].fillna(-1).astype(np.int32)
    command_features = {
        "full": time_safe_command_features(data, target, None),
        "recent": time_safe_command_features(data, target, 1),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_command_models(no_month_features, target, model_dir, "no_month", args)
    command_model_columns = {}
    for label, extra in command_features.items():
        features = pd.concat([
            no_month_features.reset_index(drop=True), extra.reset_index(drop=True),
        ], axis=1)
        fit_command_models(features, target, model_dir, label, args)
        command_model_columns[label] = list(features.columns)

    resolution_features = base_features.drop(columns=["season", "game_month"])
    context = resolution_context_frame(data)
    resolution = {}
    for mode in RESOLUTION_CONTEXTS:
        label = resolution_label(context, target, mode)
        resolution[mode] = fit_resolution_models(
            resolution_features, label, context, mode, model_dir, args,
        )

    data_with_target = data.copy()
    data_with_target[TARGET_COL] = target
    reports = audited_oof(root, data_with_target)
    target_season = int(data["season"].max()) + 1
    configuration = {
        "policy": POLICY,
        "command_blend_scale": COMMAND_BLEND_SCALE,
        "game_type_regular": "R", "game_type_futures": "F",
        "early_pitcher_pitches": 600.0,
        "pitcher_season_origins": freeze_pitcher_season_origins(data),
        "command": {
            "no_month_feature_columns": list(no_month_features.columns),
            "full_feature_columns": command_model_columns["full"],
            "recent_feature_columns": command_model_columns["recent"],
            "full_lookup": freeze_command(data, target, target_season, None),
            "recent_lookup": freeze_command(data, target, target_season, 1),
            "seeds": list(COMMAND_SEEDS),
        },
        "resolution": {
            "model_feature_columns": list(resolution_features.columns),
            "centers": resolution, "seeds": list(RESOLUTION_SEEDS),
        },
        "pressure": freeze_pressure(data, target),
        "row_independent_inference": True,
        "target_season": target_season,
    }
    metadata["version"] = "v24_robust_command_resolution"
    names = metadata.setdefault("model_names", [])
    if "v24_robust_candidate" not in names:
        names.append("v24_robust_candidate")
    metadata["v24_robust_candidate"] = configuration
    metadata["training_info"]["v24_validation"] = {
        "reports": reports,
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "forbidden_2025_trackman_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(f"v24 validation: {json.dumps(reports)}", flush=True)
    print("Stored v24 models, frozen lookups, metadata, and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
