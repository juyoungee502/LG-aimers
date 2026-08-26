"""LG Aimers Phase 2 inference entrypoint.

Every evaluation row is transformed independently using only its own values and
frozen statistics learned from train.csv.  No aggregation over test.csv occurs.
"""

from __future__ import annotations

import os
import time
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ID_COL = "row_id"
TARGET_COL = "control_success"
RATE_EPS = 1e-6


def add_season_features(
    out, prefix, career_n_col, career_rate_col,
    base_n, base_success, fallback_prior,
):
    career_n = out[career_n_col].fillna(0).to_numpy(dtype=np.float64)
    career_rate = out[career_rate_col].fillna(fallback_prior).to_numpy(dtype=np.float64)
    career_success = np.rint(career_n * career_rate)
    season_n = np.maximum(0.0, career_n - base_n)
    season_success = np.clip(career_success - base_success, 0.0, season_n)
    base_rate = np.divide(
        base_success, base_n,
        out=np.full(len(out), fallback_prior, dtype=np.float64),
        where=base_n > 0,
    )
    season_rate = np.divide(
        season_success, season_n, out=base_rate.copy(), where=season_n > 0,
    )

    out[f"{prefix}_career_success_count"] = career_success.astype(np.float32)
    out[f"{prefix}_season_n"] = season_n.astype(np.float32)
    out[f"{prefix}_season_success_count"] = season_success.astype(np.float32)
    out[f"{prefix}_season_success_rate"] = season_rate.astype(np.float32)
    out[f"{prefix}_prior_success_rate"] = base_rate.astype(np.float32)
    out[f"{prefix}_season_minus_prior"] = (season_rate - base_rate).astype(np.float32)
    out[f"{prefix}_season_log_n"] = np.log1p(season_n).astype(np.float32)
    for strength in (10.0, 25.0, 50.0, 100.0, 200.0):
        smoothed = (season_success + strength * base_rate) / (season_n + strength)
        out[f"{prefix}_season_success_s{int(strength)}"] = smoothed.astype(np.float32)
        out[f"{prefix}_season_weight_s{int(strength)}"] = (
            season_n / (season_n + strength)
        ).astype(np.float32)


def build_features(df, bundle):
    history = bundle["history"]
    p_ids = df["pitcher_id"]
    b_ids = df["batter_id"]
    p_base_n = p_ids.map(history["pitcher_n"]).fillna(0).to_numpy(np.float32)
    p_base_s = p_ids.map(history["pitcher_success"]).fillna(0).to_numpy(np.float32)
    b_base_n = b_ids.map(history["batter_n"]).fillna(0).to_numpy(np.float32)
    b_base_s = b_ids.map(history["batter_success"]).fillna(0).to_numpy(np.float32)

    out = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
    out["top_bottom"] = out["top_bottom"].map({"T": 0, "B": 1}).fillna(-1)
    out["game_type"] = out["game_type"].map({"R": 0, "F": 1}).fillna(-1)
    base_map = {
        "___": 0, "1__": 1, "_2_": 2, "__3": 3,
        "12_": 4, "1_3": 5, "_23": 6, "123": 7,
    }
    out["base_state"] = out["base_state"].map(base_map).fillna(-1)

    add_season_features(
        out, "pitcher", "asof_pitcher_n", "asof_pitcher_success_rate",
        p_base_n, p_base_s, history["global_prior"],
    )
    add_season_features(
        out, "batter", "asof_batter_n", "asof_batter_success_rate",
        b_base_n, b_base_s, history["global_prior"],
    )

    out["count_state"] = out["balls_before"] * 3 + out["strikes_before"]
    out["base_out_state"] = out["base_state"] * 3 + out["outs_before"]
    out["hand_matchup"] = out["pitcher_hand"] * 3 + out["batter_hand"]
    out["team_matchup"] = out["pitcher_team_id"] * 32 + out["batter_team_id"]
    out["is_pitcher_home"] = (
        ((out["top_bottom"] == 0)
         & (out["score_diff_pitcher_team"] == out["score_diff_home"]))
        | ((out["top_bottom"] == 1)
           & (out["score_diff_pitcher_team"] == -out["score_diff_home"]))
    ).astype(np.int8)
    out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
    out["is_tied"] = (out["score_diff_pitcher_team"] == 0).astype(np.int8)
    out["is_pitcher_ahead"] = (out["score_diff_pitcher_team"] > 0).astype(np.int8)
    out["is_late"] = (out["inning"] >= 7).astype(np.int8)
    out["is_extra_inning"] = (out["inning"] >= 10).astype(np.int8)
    out["two_strike"] = (out["strikes_before"] == 2).astype(np.int8)
    out["three_ball"] = (out["balls_before"] == 3).astype(np.int8)
    out["full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype(np.int8)
    out["runners_in_scoring_position"] = (
        out["runner_on_2b"] + out["runner_on_3b"]
    )
    out["pressure_x_runners"] = out["li"] * (1.0 + out["num_runners_on"])

    for a, b in ((1, 3), (3, 5), (1, 5)):
        out[f"pitcher_success_trend_{a}_{b}"] = (
            out[f"asof_pitcher_prev{a}_game_success_rate"]
            - out[f"asof_pitcher_prev{b}_game_success_rate"]
        )
        out[f"pitcher_middle_trend_{a}_{b}"] = (
            out[f"asof_pitcher_prev{a}_game_middle_rate"]
            - out[f"asof_pitcher_prev{b}_game_middle_rate"]
        )

    out["pitchmix_sum"] = (
        out["asof_pitcher_fastball_rate"]
        + out["asof_pitcher_breaking_rate"]
        + out["asof_pitcher_offspeed_rate"]
    )
    rates = out[[
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]].clip(lower=RATE_EPS)
    out["pitchmix_entropy"] = -(rates * np.log(rates)).sum(axis=1, min_count=1)
    out["pitchmix_max"] = rates.max(axis=1)

    for col in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))
    missing_cols = [c for c in out.columns if c.startswith("asof_")]
    out["asof_missing_count"] = out[missing_cols].isna().sum(axis=1).astype(np.int8)
    out["pitcher_cold_start"] = (out["asof_pitcher_n"] == 0).astype(np.int8)
    out["batter_cold_start"] = (out["asof_batter_n"] == 0).astype(np.int8)

    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")

    expected = bundle["feature_columns"]
    missing = [c for c in expected if c not in out.columns]
    unexpected = [c for c in out.columns if c not in expected]
    if missing or unexpected:
        raise ValueError(
            f"Feature schema mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return out.reindex(columns=expected)


def history_expert(features, prior):
    specs = [
        ("asof_pitcher_prev1_game_success_rate", .12),
        ("asof_pitcher_prev3_game_success_rate", .28),
        ("asof_pitcher_prev5_game_success_rate", .20),
        ("pitcher_season_success_s100", .22),
        ("asof_pitcher_success_rate", .10),
        ("asof_batter_success_rate", .08),
    ]
    total = np.zeros(len(features), np.float64)
    weight = np.zeros(len(features), np.float64)
    for col, component_weight in specs:
        values = features[col].to_numpy(np.float64)
        valid = np.isfinite(values)
        total[valid] += component_weight * values[valid]
        weight[valid] += component_weight
    return np.divide(
        total, weight, out=np.full(len(features), prior), where=weight > 0
    )


def main():
    started = time.time()
    # The evaluation runner may execute /app/script.py while its current working
    # directory is not /app. Resolve every competition path from this file.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    metadata_path = os.path.join(model_dir, "metadata.json")
    test_path = os.path.join(base_dir, "data", "test.csv")
    output_path = os.path.join(base_dir, "output", "submission.csv")

    print("Loading native models...")
    with open(metadata_path, "r", encoding="utf-8") as file:
        bundle = json.load(file)
    for name in ("pitcher_n", "pitcher_success", "batter_n", "batter_success"):
        bundle["history"][name] = {
            int(key): value for key, value in bundle["history"][name].items()
        }
    with open(os.path.join(model_dir, "lgb_0.txt"), "r", encoding="utf-8") as file:
        lgb_0_text = file.read()
    with open(os.path.join(model_dir, "lgb_1.txt"), "r", encoding="utf-8") as file:
        lgb_1_text = file.read()
    models = {
        "lgb_a": lgb.Booster(model_str=lgb_0_text),
        "lgb_b": lgb.Booster(model_str=lgb_1_text),
        "catboost": CatBoostClassifier(),
    }
    models["catboost"].load_model(os.path.join(model_dir, "catboost.cbm"))
    print("Model version:", bundle["version"])
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    if ID_COL not in test.columns or test[ID_COL].duplicated().any():
        raise ValueError("test.csv must contain unique row_id values")

    features = build_features(test, bundle)
    for col in bundle["cat_features"]:
        features[col] = features[col].fillna(-1).astype(np.int32)
    if list(features.columns) != bundle["feature_columns"]:
        raise ValueError("Final feature order differs from training")

    weights = bundle["blend_weights"]
    prediction = (
        weights["lgb_a"] * models["lgb_a"].predict(features)
        + weights["lgb_b"] * models["lgb_b"].predict(features)
        + weights["catboost"] * models["catboost"].predict_proba(features)[:, 1]
        + weights["history_expert"] * history_expert(
            features, bundle["history"]["global_prior"]
        )
    )
    calibration = bundle["calibration"]
    prediction = calibration["intercept"] + calibration["slope"] * prediction
    prediction = np.clip(prediction, *bundle["clip"])
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction length or non-finite prediction")
    if np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Predictions are outside [0, 1]")

    submission = pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy(copy=True),
        TARGET_COL: prediction,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Saved {output_path}: rows={len(submission)}, "
        f"mean={prediction.mean():.6f}, elapsed={time.time() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
