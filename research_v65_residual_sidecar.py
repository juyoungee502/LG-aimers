"""Strict shallow residual sidecar audit over the complete v64 OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
CATEGORICAL = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
]
SCALES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)


def parameters(seed: int, depth: int, task_type: str, devices: str) -> dict[str, object]:
    result: dict[str, object] = {
        "iterations": 750,
        "learning_rate": 0.018,
        "depth": depth,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 500.0,
        "random_strength": 2.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.7,
        "random_seed": seed,
        "task_type": task_type,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if task_type == "GPU":
        result.update(devices=devices, gpu_ram_part=0.85, border_count=32)
    return result


def centered_residual(
    target: np.ndarray,
    baseline: np.ndarray,
    game_type: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = np.asarray(target, dtype=float) - np.asarray(baseline, dtype=float)
    centered = residual.copy()
    centers: dict[str, float] = {}
    for group in ("R", "F"):
        mask = np.asarray(game_type).astype(str) == group
        center = float(residual[mask].mean())
        centered[mask] -= center
        centers[group] = center
    return centered, centers


def fit_members(
    features: pd.DataFrame,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    residual: np.ndarray,
    task_type: str,
    devices: str,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for depth in (4, 6):
        predictions: list[np.ndarray] = []
        for member in range(2):
            model = CatBoostRegressor(
                **parameters(seed_offset + depth * 100 + member * 37, depth, task_type, devices)
            )
            model.fit(
                features.iloc[fit_indices],
                residual,
                cat_features=CATEGORICAL,
            )
            predictions.append(np.asarray(model.predict(features.iloc[predict_indices]), dtype=float))
        output[f"d{depth}"] = np.mean(predictions, axis=0)
    output["mean"] = 0.5 * (output["d4"] + output["d6"])
    return output


def score_gain(target: np.ndarray, baseline: np.ndarray, prediction: np.ndarray) -> float:
    return float(bss(target, prediction) - bss(target, baseline))


def evaluate_grid(
    target: np.ndarray,
    baseline: np.ndarray,
    correction: np.ndarray,
    game_type: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    indices = np.arange(len(target))
    halves = np.array_split(indices, 2)
    quarters = np.array_split(indices, 4)
    regular = np.asarray(game_type).astype(str) == "R"
    for r_scale in SCALES:
        for f_scale in SCALES:
            scales = np.where(regular, r_scale, f_scale)
            prediction = np.clip(baseline + scales * correction, 1e-6, 1.0 - 1e-6)
            records.append(
                {
                    "r_scale": r_scale,
                    "f_scale": f_scale,
                    "gain": score_gain(target, baseline, prediction),
                    "half_gains": [score_gain(target[i], baseline[i], prediction[i]) for i in halves],
                    "quarter_gains": [score_gain(target[i], baseline[i], prediction[i]) for i in quarters],
                    "group_gains": {
                        "R": score_gain(target[regular], baseline[regular], prediction[regular]),
                        "F": score_gain(target[~regular], baseline[~regular], prediction[~regular]),
                    },
                    "mean_absolute_change": float(np.mean(np.abs(prediction - baseline))),
                }
            )
    return records


def bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    reference = float(target.mean() * (1.0 - target.mean()))
    row_gain = np.square(baseline - target) - np.square(prediction - target)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": row_gain}).groupby("pitcher")["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(dtype=float)
    sizes = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        sampled = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = 100_000.0 * sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1) / reference
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(dtype=float)
    history = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *history, global_prior=float(target_series.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in CATEGORICAL:
        features[column] = features[column].fillna(-1).astype(str)

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        target = archive["target"].astype(float)
        baseline = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    active = raw["season"].isin([2023, 2024]).to_numpy()
    if not np.array_equal(target, target_all[active]):
        raise ValueError("v64 OOF and training rows are not aligned")
    active_features = features.loc[active].reset_index(drop=True)
    active_rows = raw.loc[active].reset_index(drop=True)
    game_type = active_rows["game_type"].astype(str).to_numpy()
    index23 = np.flatnonzero(season == 2023)
    index24 = np.flatnonzero(season == 2024)
    first23, second23 = np.array_split(index23, 2)

    residual_first, centers_first = centered_residual(
        target[first23], baseline[first23], game_type[first23]
    )
    within_members = fit_members(
        active_features, first23, second23, residual_first,
        args.task_type, args.devices, 65100,
    )
    residual23, centers23 = centered_residual(
        target[index23], baseline[index23], game_type[index23]
    )
    forward_members = fit_members(
        active_features, index23, index24, residual23,
        args.task_type, args.devices, 65200,
    )

    summaries: list[dict[str, object]] = []
    for name in ("d4", "d6", "mean"):
        within_grid = evaluate_grid(
            target[second23], baseline[second23], within_members[name], game_type[second23]
        )
        forward_grid = evaluate_grid(
            target[index24], baseline[index24], forward_members[name], game_type[index24]
        )
        by_key_within = {(row["r_scale"], row["f_scale"]): row for row in within_grid}
        by_key_forward = {(row["r_scale"], row["f_scale"]): row for row in forward_grid}
        candidates: list[dict[str, object]] = []
        for key in by_key_within:
            within = by_key_within[key]
            forward = by_key_forward[key]
            scales = np.where(game_type[index24] == "R", key[0], key[1])
            prediction = np.clip(
                baseline[index24] + scales * forward_members[name], 1e-6, 1.0 - 1e-6
            )
            clustered = bootstrap(
                target[index24], baseline[index24], prediction,
                active_rows.loc[index24, "pitcher_id"].to_numpy(),
                args.bootstrap, 65300 + int(1000 * key[0]) + int(10000 * key[1]),
            )
            strict_gate = bool(
                within["gain"] > 0.0
                and forward["gain"] > 0.0
                and min(within["half_gains"]) >= 0.0
                and min(forward["half_gains"]) >= 0.0
                and min(forward["group_gains"].values()) >= 0.0
                and clustered["ci_low"] > 0.0
            )
            candidates.append(
                {
                    "r_scale": key[0],
                    "f_scale": key[1],
                    "within_2023_second_half": within,
                    "forward_2024": forward,
                    "bootstrap_2024": clustered,
                    "strict_gate": strict_gate,
                }
            )
        candidates.sort(
            key=lambda row: (
                row["strict_gate"],
                min(row["within_2023_second_half"]["gain"], row["forward_2024"]["gain"]),
                row["forward_2024"]["gain"],
            ),
            reverse=True,
        )
        summaries.append(
            {
                "model": name,
                "best": candidates[0],
                "selected": candidates[0] if candidates[0]["strict_gate"] else None,
                "correction_std_2024": float(forward_members[name].std()),
            }
        )

    summaries.sort(
        key=lambda row: (
            row["selected"] is not None,
            min(row["best"]["within_2023_second_half"]["gain"], row["best"]["forward_2024"]["gain"]),
        ),
        reverse=True,
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "candidate_family": "shallow_full_feature_residual_sidecar",
        "source_centers_removed": {
            "within_2023_first_half": centers_first,
            "forward_2024_full_2023": centers23,
        },
        "models": summaries,
        "selected": summaries[0] if summaries[0]["selected"] is not None else None,
        "rules": {
            "official_data_only": True,
            "validation_or_test_peer_aggregation": False,
            "forbidden_2025_trackman_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "research/v65_residual_sidecar.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "outputs/v65_residual_sidecar_oof.npz",
        target_2024=target[index24],
        baseline_2024=baseline[index24],
        game_type_2024=game_type[index24],
        pitcher_2024=active_rows.loc[index24, "pitcher_id"].to_numpy(),
        correction_d4=forward_members["d4"],
        correction_d6=forward_members["d6"],
        correction_mean=forward_members["mean"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
