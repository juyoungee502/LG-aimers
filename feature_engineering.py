"""Time-safe, row-independent features for the LG Aimers pitching task.

The test transformer only combines values from the current test row with frozen
end-of-2024 statistics learned from train.csv.  It never groups, sorts, or
otherwise inspects other test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"

RATE_EPS = 1e-6

COMPONENT_SPECS = (
    ("pitcher_reverse", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate"),
    ("pitcher_middle", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate"),
    ("pitcher_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate"),
    ("pitcher_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate"),
    ("pitcher_fastball", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate"),
    ("pitcher_breaking", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_breaking_rate"),
    ("pitcher_offspeed", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_offspeed_rate"),
    ("batter_middle", "batter_id", "asof_batter_n", "asof_batter_middle_rate"),
)


@dataclass
class HistoryTables:
    global_prior: float
    pitcher_n: dict[int, float]
    pitcher_success: dict[int, float]
    batter_n: dict[int, float]
    batter_success: dict[int, float]
    components: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)


def _success_count(n: pd.Series, rate: pd.Series) -> np.ndarray:
    """Recover the integer count represented by a cumulative count/rate pair."""
    n_values = n.fillna(0).to_numpy(dtype=np.float64, copy=False)
    r_values = rate.fillna(0).to_numpy(dtype=np.float64, copy=False)
    return np.rint(n_values * r_values)


def _component_end_maps(
    train: pd.DataFrame, id_col: str, n_col: str, rate_col: str
) -> tuple[dict[int, float], dict[int, float]]:
    last = train.groupby(id_col, sort=False, observed=True).tail(1)
    ids = last[id_col].astype(np.int64).to_numpy()
    n = last[n_col].fillna(0).to_numpy(np.float64)
    rate = last[rate_col].fillna(0).to_numpy(np.float64)
    # The final pitch's component label is not present.  Add its pre-pitch rate
    # as an unbiased fractional expectation; this limits the uncertainty to one
    # event per player and avoids consulting any evaluation row.
    end_n = n + 1.0
    end_count = np.rint(n * rate) + rate
    return dict(zip(ids.tolist(), end_n.tolist())), dict(zip(ids.tolist(), end_count.tolist()))


def build_end_history(train: pd.DataFrame, target: pd.Series) -> HistoryTables:
    """Build frozen player totals immediately after each player's final train row."""
    work = train[[
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    ]].copy()
    work["_target"] = target.to_numpy(dtype=np.float64, copy=False)

    p_last = work.groupby("pitcher_id", sort=False, observed=True).tail(1)
    p_before = _success_count(
        p_last["asof_pitcher_n"], p_last["asof_pitcher_success_rate"]
    )
    p_ids = p_last["pitcher_id"].astype(np.int64).to_numpy()
    p_n = p_last["asof_pitcher_n"].to_numpy(dtype=np.float64) + 1.0
    p_s = p_before + p_last["_target"].to_numpy(dtype=np.float64)

    b_last = work.groupby("batter_id", sort=False, observed=True).tail(1)
    b_before = _success_count(
        b_last["asof_batter_n"], b_last["asof_batter_success_rate"]
    )
    b_ids = b_last["batter_id"].astype(np.int64).to_numpy()
    b_n = b_last["asof_batter_n"].to_numpy(dtype=np.float64) + 1.0
    b_s = b_before + b_last["_target"].to_numpy(dtype=np.float64)

    components = {}
    for name, id_col, n_col, rate_col in COMPONENT_SPECS:
        component_n, component_count = _component_end_maps(train, id_col, n_col, rate_col)
        components[name] = {"n": component_n, "count": component_count}

    return HistoryTables(
        global_prior=float(target.mean()),
        pitcher_n=dict(zip(p_ids.tolist(), p_n.tolist())),
        pitcher_success=dict(zip(p_ids.tolist(), p_s.tolist())),
        batter_n=dict(zip(b_ids.tolist(), b_n.tolist())),
        batter_success=dict(zip(b_ids.tolist(), b_s.tolist())),
        components=components,
    )


def _training_component_arrays(
    train: pd.DataFrame, id_col: str, n_col: str, rate_col: str
) -> tuple[np.ndarray, np.ndarray]:
    size = len(train)
    base_n = np.zeros(size, dtype=np.float32)
    base_count = np.zeros(size, dtype=np.float32)
    n_map: dict[int, float] = {}
    count_map: dict[int, float] = {}
    seasons = train["season"].to_numpy()

    for season in np.sort(train["season"].unique()):
        positions = np.flatnonzero(seasons == season)
        block = train.iloc[positions]
        ids = block[id_col]
        base_n[positions] = ids.map(n_map).fillna(0).to_numpy(np.float32)
        base_count[positions] = ids.map(count_map).fillna(0).to_numpy(np.float32)

        last_positions = block.groupby(id_col, sort=False).tail(1).index
        last = train.loc[last_positions]
        last_ids = last[id_col].astype(int).to_numpy()
        n = last[n_col].fillna(0).to_numpy(np.float64)
        rate = last[rate_col].fillna(0).to_numpy(np.float64)
        n_map.update(zip(last_ids, n + 1.0))
        count_map.update(zip(last_ids, np.rint(n * rate) + rate))

    return base_n, base_count


def add_training_component_features(out: pd.DataFrame, raw: pd.DataFrame) -> None:
    for name, id_col, n_col, rate_col in COMPONENT_SPECS:
        base_n, base_count = _training_component_arrays(raw, id_col, n_col, rate_col)
        _add_component_features(out, name, n_col, rate_col, base_n, base_count)


def add_inference_component_features(
    out: pd.DataFrame, raw: pd.DataFrame, history: HistoryTables
) -> None:
    for name, id_col, n_col, rate_col in COMPONENT_SPECS:
        maps = history.components[name]
        base_n = raw[id_col].map(maps["n"]).fillna(0).to_numpy(np.float32)
        base_count = raw[id_col].map(maps["count"]).fillna(0).to_numpy(np.float32)
        _add_component_features(out, name, n_col, rate_col, base_n, base_count)


def _add_component_features(
    out: pd.DataFrame,
    name: str,
    n_col: str,
    rate_col: str,
    base_n: np.ndarray,
    base_count: np.ndarray,
) -> None:
    n = out[n_col].fillna(0).to_numpy(np.float64)
    rate = out[rate_col].fillna(0).to_numpy(np.float64)
    count = np.rint(n * rate)
    season_n = np.maximum(0.0, n - base_n)
    season_count = np.clip(count - base_count, 0.0, season_n)
    prior_rate = np.divide(
        base_count, base_n, out=rate.copy(), where=base_n > 0
    )
    season_rate = np.divide(
        season_count, season_n, out=prior_rate.copy(), where=season_n > 0
    )
    out[f"{name}_season_rate"] = season_rate.astype(np.float32)
    out[f"{name}_season_s25"] = (
        (season_count + 25.0 * prior_rate) / (season_n + 25.0)
    ).astype(np.float32)
    out[f"{name}_season_s100"] = (
        (season_count + 100.0 * prior_rate) / (season_n + 100.0)
    ).astype(np.float32)
    out[f"{name}_season_delta"] = (season_rate - prior_rate).astype(np.float32)


def training_history_arrays(
    train: pd.DataFrame, target: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return prior-season frozen totals for every training row.

    For rows in season Y, only rows from seasons before Y are used.  The current
    season's labels update the frozen maps only after all rows in that season
    have received their feature values.
    """
    size = len(train)
    p_base_n = np.zeros(size, dtype=np.float32)
    p_base_s = np.zeros(size, dtype=np.float32)
    b_base_n = np.zeros(size, dtype=np.float32)
    b_base_s = np.zeros(size, dtype=np.float32)

    p_n_map: dict[int, float] = {}
    p_s_map: dict[int, float] = {}
    b_n_map: dict[int, float] = {}
    b_s_map: dict[int, float] = {}

    target_values = target.to_numpy(dtype=np.float64, copy=False)
    seasons = train["season"].to_numpy()

    for season in np.sort(train["season"].unique()):
        positions = np.flatnonzero(seasons == season)
        block = train.iloc[positions]

        p_ids = block["pitcher_id"]
        b_ids = block["batter_id"]
        p_base_n[positions] = p_ids.map(p_n_map).fillna(0).to_numpy(np.float32)
        p_base_s[positions] = p_ids.map(p_s_map).fillna(0).to_numpy(np.float32)
        b_base_n[positions] = b_ids.map(b_n_map).fillna(0).to_numpy(np.float32)
        b_base_s[positions] = b_ids.map(b_s_map).fillna(0).to_numpy(np.float32)

        p_last_positions = block.groupby("pitcher_id", sort=False).tail(1).index
        p_last = train.loc[p_last_positions]
        p_end_n = p_last["asof_pitcher_n"].to_numpy(np.float64) + 1.0
        p_end_s = _success_count(
            p_last["asof_pitcher_n"], p_last["asof_pitcher_success_rate"]
        ) + target_values[p_last_positions]
        p_n_map.update(zip(p_last["pitcher_id"].astype(int), p_end_n))
        p_s_map.update(zip(p_last["pitcher_id"].astype(int), p_end_s))

        b_last_positions = block.groupby("batter_id", sort=False).tail(1).index
        b_last = train.loc[b_last_positions]
        b_end_n = b_last["asof_batter_n"].to_numpy(np.float64) + 1.0
        b_end_s = _success_count(
            b_last["asof_batter_n"], b_last["asof_batter_success_rate"]
        ) + target_values[b_last_positions]
        b_n_map.update(zip(b_last["batter_id"].astype(int), b_end_n))
        b_s_map.update(zip(b_last["batter_id"].astype(int), b_end_s))

    return p_base_n, p_base_s, b_base_n, b_base_s


def inference_history_arrays(
    df: pd.DataFrame, history: HistoryTables
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map frozen training constants to each row independently."""
    p_ids = df["pitcher_id"]
    b_ids = df["batter_id"]
    return (
        p_ids.map(history.pitcher_n).fillna(0).to_numpy(np.float32),
        p_ids.map(history.pitcher_success).fillna(0).to_numpy(np.float32),
        b_ids.map(history.batter_n).fillna(0).to_numpy(np.float32),
        b_ids.map(history.batter_success).fillna(0).to_numpy(np.float32),
    )


def _add_season_features(
    out: pd.DataFrame,
    prefix: str,
    career_n_col: str,
    career_rate_col: str,
    base_n: np.ndarray,
    base_success: np.ndarray,
    fallback_prior: float,
) -> None:
    career_n = out[career_n_col].fillna(0).to_numpy(dtype=np.float64)
    career_rate = out[career_rate_col].fillna(fallback_prior).to_numpy(dtype=np.float64)
    career_success = np.rint(career_n * career_rate)

    season_n = np.maximum(0.0, career_n - base_n)
    season_success = np.clip(career_success - base_success, 0.0, season_n)
    base_rate = np.divide(
        base_success,
        base_n,
        out=np.full(len(out), fallback_prior, dtype=np.float64),
        where=base_n > 0,
    )
    season_rate = np.divide(
        season_success,
        season_n,
        out=base_rate.copy(),
        where=season_n > 0,
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


def engineer_features(
    df: pd.DataFrame,
    p_base_n: np.ndarray,
    p_base_s: np.ndarray,
    b_base_n: np.ndarray,
    b_base_s: np.ndarray,
    global_prior: float,
) -> pd.DataFrame:
    """Create model features without reading any other evaluation row."""
    out = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()

    out["top_bottom"] = out["top_bottom"].map({"T": 0, "B": 1}).fillna(-1)
    out["game_type"] = out["game_type"].map({"R": 0, "F": 1}).fillna(-1)
    base_map = {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                "12_": 4, "1_3": 5, "_23": 6, "123": 7}
    out["base_state"] = out["base_state"].map(base_map).fillna(-1)

    _add_season_features(
        out, "pitcher", "asof_pitcher_n", "asof_pitcher_success_rate",
        p_base_n, p_base_s, global_prior,
    )
    _add_season_features(
        out, "batter", "asof_batter_n", "asof_batter_success_rate",
        b_base_n, b_base_s, global_prior,
    )

    out["count_state"] = out["balls_before"] * 3 + out["strikes_before"]
    out["base_out_state"] = out["base_state"] * 3 + out["outs_before"]
    out["hand_matchup"] = out["pitcher_hand"] * 3 + out["batter_hand"]
    out["team_matchup"] = out["pitcher_team_id"] * 32 + out["batter_team_id"]
    out["is_pitcher_home"] = (
        ((out["top_bottom"] == 0) & (out["score_diff_pitcher_team"] == out["score_diff_home"]))
        | ((out["top_bottom"] == 1) & (out["score_diff_pitcher_team"] == -out["score_diff_home"]))
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

    for window_a, window_b in ((1, 3), (3, 5), (1, 5)):
        out[f"pitcher_success_trend_{window_a}_{window_b}"] = (
            out[f"asof_pitcher_prev{window_a}_game_success_rate"]
            - out[f"asof_pitcher_prev{window_b}_game_success_rate"]
        )
        out[f"pitcher_middle_trend_{window_a}_{window_b}"] = (
            out[f"asof_pitcher_prev{window_a}_game_middle_rate"]
            - out[f"asof_pitcher_prev{window_b}_game_middle_rate"]
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

    count_cols = [
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"
    ]
    for col in count_cols:
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

    return out
