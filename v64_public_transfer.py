"""Row-local inference helpers for the v64 public-method transfer."""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-6
F_CATEGORICAL = (
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "game_dayofweek", "count_state",
    "hand_matchup", "prior_game_type", "league_transition", "team_type",
)
F_DROP_COLUMNS = ("row_id", "control_success", "pitcher_id", "batter_id")


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def expit(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -30.0, 30.0)))


def build_f_features(
    rows: pd.DataFrame,
    base_prediction: np.ndarray,
    prior_game_type: dict[int, str],
) -> pd.DataFrame:
    """Build the F residual design without aggregating evaluation rows."""
    features = rows.drop(columns=list(F_DROP_COLUMNS), errors="ignore").copy()
    prior = rows["pitcher_id"].map(prior_game_type).fillna("NEW")
    features["prior_game_type"] = prior
    features["league_transition"] = (
        prior.astype(str) + ">" + rows["game_type"].astype(str)
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
    features["base_prediction"] = np.asarray(base_prediction, dtype=np.float32)
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
    for column in F_CATEGORICAL:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    return features


def apply_dynamic_pitcher_state(
    rows: pd.DataFrame,
    configuration: dict,
) -> np.ndarray:
    """Apply a frozen prior-season AR state using only each row's as-of fields."""
    ids = rows["pitcher_id"]
    prior_n_map = {int(key): float(value) for key, value in configuration["prior_n"].items()}
    prior_success_map = {
        int(key): float(value) for key, value in configuration["prior_success"].items()
    }
    latent_map = {int(key): float(value) for key, value in configuration["latent"].items()}
    year_map = {int(key): float(value) for key, value in configuration["latent_year"].items()}
    prior_n = ids.map(prior_n_map).fillna(0.0).to_numpy(float)
    prior_success = ids.map(prior_success_map).fillna(0.0).to_numpy(float)
    career_n = rows["asof_pitcher_n"].fillna(0).to_numpy(float)
    career_rate = rows["asof_pitcher_success_rate"].fillna(
        float(configuration["league_prior"])
    ).to_numpy(float)
    career_success = np.rint(career_n * career_rate)
    current_n = np.maximum(career_n - prior_n, 0.0)
    current_success = np.clip(career_success - prior_success, 0.0, current_n)

    latest_latent = ids.map(latent_map).fillna(0.0).to_numpy(float)
    latest_year = ids.map(year_map).to_numpy(float)
    known = np.isfinite(latest_year)
    gap = np.where(known, float(configuration["prediction_year"]) - latest_year, 0.0)
    ar_latent = np.where(
        known,
        latest_latent * np.power(float(configuration["rho"]), gap),
        0.0,
    )
    league_prior = float(configuration["league_prior"])
    state_prior = expit(logit(league_prior) + ar_latent)
    strength = float(configuration["current_prior_strength"])
    dynamic = (current_success + strength * state_prior) / (current_n + strength)
    neutral = (current_success + strength * league_prior) / (current_n + strength)
    correction = float(configuration["weight"]) * (dynamic - neutral)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    return correction * regular.astype(float)
