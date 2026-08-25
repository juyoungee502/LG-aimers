"""Leakage-safe feature engineering shared by training and inference."""
from __future__ import annotations
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLUMNS = [
    "season_cat", "game_month_cat", "game_dayofweek_cat", "inning_cat",
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "count_state", "hand_matchup", "pitcher_team_matchup",
]
RATE_PRIORS = {
    "asof_pitcher_success_rate": ("asof_pitcher_n", 80.0, 0.52),
    "asof_pitcher_reverse_rate": ("asof_pitcher_n", 80.0, 0.27),
    "asof_pitcher_middle_rate": ("asof_pitcher_n", 80.0, 0.17),
    "asof_pitcher_ball_rate": ("asof_pitcher_n", 80.0, 0.40),
    "asof_pitcher_strike_rate": ("asof_pitcher_n", 80.0, 0.40),
    "asof_batter_success_rate": ("asof_batter_n", 100.0, 0.52),
    "asof_batter_middle_rate": ("asof_batter_n", 100.0, 0.17),
    "asof_pitcher_fastball_rate": ("asof_pitcher_pitchmix_n", 60.0, 0.50),
    "asof_pitcher_breaking_rate": ("asof_pitcher_pitchmix_n", 60.0, 0.35),
    "asof_pitcher_offspeed_rate": ("asof_pitcher_pitchmix_n", 60.0, 0.15),
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
    for col in ("season", "game_month", "game_dayofweek", "inning"):
        x[f"{col}_cat"] = x[col].astype("string")
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)
    x["pitcher_team_matchup"] = x["pitcher_team_id"].astype(str) + "_" + x["batter_team_id"].astype(str)
    x["log_li"] = np.log1p(x["li"].clip(lower=0))
    x["abs_score_diff"] = x["score_diff_pitcher_team"].abs()
    x["pitcher_is_ahead"] = (x["score_diff_pitcher_team"] > 0).astype(np.int8)
    x["two_strikes"] = (x["strikes_before"] == 2).astype(np.int8)
    x["three_balls"] = (x["balls_before"] == 3).astype(np.int8)
    x["full_count"] = ((x["balls_before"] == 3) & (x["strikes_before"] == 2)).astype(np.int8)
    x["runners_scoring_position"] = ((x["runner_on_2b"] == 1) | (x["runner_on_3b"] == 1)).astype(np.int8)
    for ncol in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        x[f"log1p_{ncol}"] = np.log1p(x[ncol].clip(lower=0))
        x[f"cold_{ncol}"] = (x[ncol] == 0).astype(np.int8)
    for rate, (ncol, alpha, prior) in RATE_PRIORS.items():
        n = x[ncol].fillna(0).clip(lower=0)
        r = x[rate].fillna(prior)
        x[f"shrunk_{rate}"] = (n * r + alpha * prior) / (n + alpha)
        x[f"missing_{rate}"] = x[rate].isna().astype(np.int8)
    x["success_recent1_delta"] = x["asof_pitcher_prev1_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["success_recent3_delta"] = x["asof_pitcher_prev3_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["success_momentum_1v5"] = x["asof_pitcher_prev1_game_success_rate"] - x["asof_pitcher_prev5_game_success_rate"]
    x["middle_momentum_1v5"] = x["asof_pitcher_prev1_game_middle_rate"] - x["asof_pitcher_prev5_game_middle_rate"]
    mix = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].fillna(0).clip(1e-8, 1)
    x["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    x["pitchmix_max"] = mix.max(axis=1)
    for col in CAT_COLUMNS:
        x[col] = x[col].astype("string").fillna("__MISSING__").astype(str)
    return x
