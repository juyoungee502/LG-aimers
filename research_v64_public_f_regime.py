"""Rebuild a public high-scoring Futures residual idea over the v61 OOF.

The public implementation improved its leaderboard score by training residual
experts only on ``game_type=F`` and by exposing a pitcher's previous dominant
league as a row-local transition feature.  This audit does not import any
external predictions or models.  It trains fresh CatBoost regressors on this
project's official data and strict v61 OOF residuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
CLIP = (0.005, 0.995)
SEEDS = (6401, 6502, 6603)
SCALES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)
CATEGORICAL = (
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "game_dayofweek", "count_state",
    "hand_matchup", "prior_game_type", "league_transition", "team_type",
)
DROP_COLUMNS = ("row_id", "control_success", "pitcher_id", "batter_id")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def prior_game_type_table(history: pd.DataFrame, year: int) -> pd.Series:
    earlier = history.loc[history["season"].lt(year)]
    counts = earlier.groupby(
        ["pitcher_id", "season", "game_type"], sort=False, observed=True,
    ).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(
        ["pitcher_id", "season"], sort=False,
    ).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", sort=False).tail(1)
    return latest.set_index("pitcher_id")["game_type"]


def build_features(
    rows: pd.DataFrame,
    base: np.ndarray,
    history: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    features = rows.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    prior = rows["pitcher_id"].map(prior_game_type_table(history, year))
    features["prior_game_type"] = prior.fillna("NEW")
    features["league_transition"] = (
        features["prior_game_type"].astype(str) + ">" + rows["game_type"].astype(str)
    )
    features["team_type"] = (
        rows["pitcher_team_id"].astype(str) + "|" + rows["game_type"].astype(str)
    )
    features["count_state"] = (
        rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str)
    )
    features["hand_matchup"] = (
        rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str)
    )
    features["base_prediction"] = np.asarray(base, dtype=np.float32)
    features["log_pitcher_n"] = np.log1p(
        rows["asof_pitcher_n"].fillna(0).clip(lower=0)
    ).astype(np.float32)
    features["log_batter_n"] = np.log1p(
        rows["asof_batter_n"].fillna(0).clip(lower=0)
    ).astype(np.float32)
    features["recent_1_minus_5"] = (
        rows["asof_pitcher_prev1_game_success_rate"]
        - rows["asof_pitcher_prev5_game_success_rate"]
    ).astype(np.float32)
    features["middle_1_minus_5"] = (
        rows["asof_pitcher_prev1_game_middle_rate"]
        - rows["asof_pitcher_prev5_game_middle_rate"]
    ).astype(np.float32)
    for column in CATEGORICAL:
        features[column] = features[column].astype("string").fillna("__MISSING__").astype(str)
    return features


def parameters(args: argparse.Namespace, seed: int) -> dict:
    result = dict(
        iterations=350,
        depth=6,
        learning_rate=0.025,
        loss_function="RMSE",
        l2_leaf_reg=100.0,
        random_strength=0.3,
        bootstrap_type="Bernoulli",
        subsample=0.80,
        random_seed=seed,
        task_type=args.task_type,
        thread_count=args.threads,
        allow_writing_files=False,
        verbose=100,
    )
    if args.task_type == "GPU":
        result.update(devices=args.devices, border_count=32, gpu_ram_part=0.85)
    return result


def fit_predict(
    train_x: pd.DataFrame,
    train_target: np.ndarray,
    valid_x: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]]:
    # Remove the source residual level.  The candidate must add conditional
    # resolution, not reproduce v63's global probability offset.
    centered = np.asarray(train_target, dtype=float) - float(np.mean(train_target))
    predictions = []
    centers = []
    for seed in SEEDS:
        model = CatBoostRegressor(**parameters(args, seed))
        model.fit(train_x, centered, cat_features=list(CATEGORICAL))
        source_prediction = model.predict(train_x)
        center = float(source_prediction.mean())
        centers.append(center)
        predictions.append(model.predict(valid_x) - center)
    return np.mean(predictions, axis=0), centers


def gain(target: np.ndarray, base: np.ndarray, correction: np.ndarray) -> float:
    prediction = np.clip(base + correction, *CLIP)
    return float(bss(target, prediction) - bss(target, base))


def scale_report(
    target: np.ndarray,
    base: np.ndarray,
    correction: np.ndarray,
    *,
    full_target: np.ndarray | None = None,
    full_base: np.ndarray | None = None,
    full_mask: np.ndarray | None = None,
) -> list[dict]:
    quarters = np.array_split(np.arange(len(target)), 4)
    output = []
    for scale in SCALES:
        delta = scale * correction
        row = {
            "scale": scale,
            "gain": gain(target, base, delta),
            "shape_only_diagnostic_gain": gain(
                target, base, scale * (correction - correction.mean()),
            ),
            "quarter_gains": [
                gain(target[index], base[index], delta[index]) for index in quarters
            ],
            "correction_mean": float(delta.mean()),
            "correction_std": float(delta.std()),
        }
        if full_target is not None and full_base is not None and full_mask is not None:
            full_delta = np.zeros(len(full_target), dtype=float)
            full_delta[full_mask] = delta
            row["full_season_gain"] = gain(full_target, full_base, full_delta)
        output.append(row)
    return output


def main() -> None:
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = oof["season"].astype(int)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(seasons) or not np.array_equal(
        rows["season"].to_numpy(int), seasons,
    ):
        raise ValueError("v61 OOF rows are not aligned with train.csv")
    target = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    features = {}
    for year in (2023, 2024):
        active = seasons == year
        features[year] = build_features(rows.loc[active].reset_index(drop=True), base[active], raw, year)

    f23 = rows.loc[seasons == 2023, "game_type"].eq("F").to_numpy()
    f24 = rows.loc[seasons == 2024, "game_type"].eq("F").to_numpy()
    y23, b23 = target[seasons == 2023][f23], base[seasons == 2023][f23]
    y24, b24 = target[seasons == 2024][f24], base[seasons == 2024][f24]
    x23 = features[2023].loc[f23].reset_index(drop=True)
    x24 = features[2024].loc[f24].reset_index(drop=True)

    split = len(x23) // 2
    within_correction, within_centers = fit_predict(
        x23.iloc[:split], y23[:split] - b23[:split], x23.iloc[split:], args,
    )
    forward_correction, forward_centers = fit_predict(
        x23, y23 - b23, x24, args,
    )
    y24_all, b24_all = target[seasons == 2024], base[seasons == 2024]
    report = {
        "baseline": "v61_public_complete_shape",
        "candidate_family": "public_rebuilt_f_residual_transition",
        "rows": {"f_2023": len(x23), "f_2024": len(x24)},
        "within_2023_second_half": scale_report(
            y23[split:], b23[split:], within_correction,
        ),
        "forward_2023_to_2024": scale_report(
            y24, b24, forward_correction,
            full_target=y24_all, full_base=b24_all, full_mask=f24,
        ),
        "model_source_prediction_centers": {
            "within": within_centers, "forward": forward_centers,
        },
        "rules": {
            "official_data_only": True,
            "external_prediction_or_model_used": False,
            "strict_prior_year_transition_table": True,
            "validation_or_test_peer_aggregation": False,
            "forbidden_2025_trackman_used": False,
            "v62_or_v63_component_used": False,
            "source_residual_level_removed": True,
        },
    }
    path = ROOT / "research/v64_public_f_regime.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
