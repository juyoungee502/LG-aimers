"""Leakage-safe hierarchical residual features for the v66 reference model.

The implementation is intentionally self contained.  It borrows the modelling
idea (not external weights or predictions) of placing a boosted residual model
around a reliability-shrunk pitcher estimate.  Every table used at inference is
frozen from labelled seasons before the prediction season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CLIP = (0.005, 0.995)

BASE_CATEGORICAL = (
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
    "count_state", "hand_matchup", "count_base_context",
    "hand_count_context", "inning_score_context", "pressure_context",
    "recent_regime",
)

RESIDUAL_CATEGORICAL = (
    "top_bottom", "base_state", "pitcher_hand", "batter_hand",
    "count_state", "hand_matchup",
)

PITCHER_RATES = (
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)
BATTER_RATES = (
    "asof_batter_success_rate", "asof_batter_middle_rate",
)
PITCHMIX_RATES = (
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


@dataclass
class SeasonSnapshots:
    pitcher: pd.DataFrame
    batter: pd.DataFrame
    pitchmix: pd.DataFrame


def _expanded_snapshot_table(
    history: pd.DataFrame,
    entity_col: str,
    n_col: str,
    rate_cols: tuple[str, ...],
    final_success_col: str | None,
) -> pd.DataFrame:
    """Capture each season-end cumulative state and carry it into later years."""
    records: list[dict[str, float | int | str]] = []
    for (entity, year), group in history.groupby(
        [entity_col, "season"], sort=False, observed=True,
    ):
        final = group.iloc[-1]
        exposure = float(final[n_col]) if pd.notna(final[n_col]) else 0.0
        record: dict[str, float | int | str] = {
            entity_col: str(entity),
            "season": int(year) + 1,
            "prior_n": exposure + 1.0,
        }
        for index, rate_col in enumerate(rate_cols):
            rate = float(final[rate_col]) if pd.notna(final[rate_col]) else 0.0
            count = exposure * rate
            if index == 0 and final_success_col is not None:
                count += float(final[final_success_col])
            record[f"prior_count_{rate_col}"] = count
        records.append(record)
    direct = pd.DataFrame(records)
    if direct.empty:
        columns = [entity_col, "season", "prior_n"] + [
            f"prior_count_{column}" for column in rate_cols
        ]
        return pd.DataFrame(columns=columns)

    last_target_year = int(history["season"].max()) + 1
    completed: list[pd.DataFrame] = []
    for entity, group in direct.groupby(entity_col, sort=False, observed=True):
        years = pd.Index(
            range(int(group["season"].min()), last_target_year + 1),
            name="season",
        )
        block = group.set_index("season").reindex(years).ffill().reset_index()
        block[entity_col] = str(entity)
        completed.append(block)
    return pd.concat(completed, ignore_index=True)


def build_snapshots(history: pd.DataFrame) -> SeasonSnapshots:
    """Build three prior-season tables from labelled historical rows only."""
    return SeasonSnapshots(
        pitcher=_expanded_snapshot_table(
            history, "pitcher_id", "asof_pitcher_n", PITCHER_RATES,
            TARGET_COL,
        ),
        batter=_expanded_snapshot_table(
            history, "batter_id", "asof_batter_n", BATTER_RATES,
            TARGET_COL,
        ),
        pitchmix=_expanded_snapshot_table(
            history, "pitcher_id", "asof_pitcher_pitchmix_n",
            PITCHMIX_RATES, None,
        ),
    )


def _lookup_snapshot(
    raw: pd.DataFrame,
    table: pd.DataFrame,
    entity_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    keys = pd.MultiIndex.from_arrays([
        raw[entity_col].astype(str), raw["season"].astype(int),
    ])
    indexed = table.set_index([entity_col, "season"])
    prior_n = indexed["prior_n"].reindex(keys).fillna(0.0).to_numpy(float)
    return prior_n, indexed


def _attach_current_season(
    features: pd.DataFrame,
    raw: pd.DataFrame,
    table: pd.DataFrame,
    entity_col: str,
    n_col: str,
    rate_cols: tuple[str, ...],
    prefix: str,
    success_prior: float,
) -> pd.DataFrame:
    """Recover within-season exposure/rates from cumulative as-of columns."""
    output = features.copy()
    prior_n, indexed = _lookup_snapshot(raw, table, entity_col)
    keys = pd.MultiIndex.from_arrays([
        raw[entity_col].astype(str), raw["season"].astype(int),
    ])
    total_n = pd.to_numeric(raw[n_col], errors="coerce").fillna(0).to_numpy(float)
    season_n = np.maximum(total_n - prior_n, 0.0)
    output[f"{prefix}_n"] = season_n.astype(np.float32)
    for rate_col in rate_cols:
        fallback = success_prior if "success" in rate_col else 0.0
        rate = pd.to_numeric(raw[rate_col], errors="coerce").fillna(fallback)
        total_count = total_n * rate.to_numpy(float)
        previous = indexed[f"prior_count_{rate_col}"].reindex(keys)
        previous_count = previous.fillna(0.0).to_numpy(float)
        season_count = np.maximum(total_count - previous_count, 0.0)
        season_rate = np.divide(
            season_count, season_n,
            out=np.full(len(raw), fallback, dtype=float),
            where=season_n > 0.0,
        )
        short_name = rate_col.removeprefix("asof_pitcher_").removeprefix(
            "asof_batter_"
        )
        output[f"{prefix}_{short_name}_raw"] = season_rate.astype(np.float32)
    return output


def _row_features(raw: pd.DataFrame, prior: float) -> pd.DataFrame:
    output = raw.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
    output["count_pressure"] = output["balls_before"] - output["strikes_before"]
    output["two_strike"] = (output["strikes_before"] == 2).astype(np.int8)
    output["three_ball"] = (output["balls_before"] == 3).astype(np.int8)
    output["late_inning"] = (output["inning"] >= 7).astype(np.int8)
    output["scoring_position"] = (
        output["runner_on_2b"].eq(1) | output["runner_on_3b"].eq(1)
    ).astype(np.int8)
    output["runners_x_li"] = output["num_runners_on"] * output["li"]
    recent = output[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]]
    output["recent_success_mean"] = recent.mean(axis=1)
    output["recent_vs_career"] = (
        output["recent_success_mean"] - output["asof_pitcher_success_rate"]
    )
    for name, n_col, rate_col in (
        ("pitcher", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "asof_batter_n", "asof_batter_success_rate"),
    ):
        n = pd.to_numeric(output[n_col], errors="coerce").fillna(0).clip(lower=0)
        rate = pd.to_numeric(output[rate_col], errors="coerce").fillna(prior)
        output[f"{name}_success_smoothed"] = (
            rate * n + prior * 100.0
        ) / (n + 100.0)
    output["count_state"] = (
        output["balls_before"].astype(str) + "-"
        + output["strikes_before"].astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype(str) + "-"
        + output["batter_hand"].astype(str)
    )
    output["score_abs"] = output["score_diff_pitcher_team"].abs()
    output["close_game"] = output["score_abs"].le(1).astype(np.int8)
    output["pressure_index"] = (
        (output["balls_before"] + 1.0) / (output["strikes_before"] + 1.0)
        * (1.0 + output["num_runners_on"])
        * np.log1p(output["li"].clip(lower=0))
    )
    output["pitcher_uncertainty"] = 1.0 / np.sqrt(
        output["asof_pitcher_n"].clip(lower=0) + 1.0
    )
    output["batter_uncertainty"] = 1.0 / np.sqrt(
        output["asof_batter_n"].clip(lower=0) + 1.0
    )
    output["recent_slope_1_5"] = (
        output["asof_pitcher_prev1_game_success_rate"]
        - output["asof_pitcher_prev5_game_success_rate"]
    )
    output["middle_slope_1_5"] = (
        output["asof_pitcher_prev1_game_middle_rate"]
        - output["asof_pitcher_prev5_game_middle_rate"]
    )
    rates = output[[
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]].clip(lower=1e-7)
    output["pitchmix_entropy"] = -(rates * np.log(rates)).sum(axis=1, min_count=1)
    return output


def build_features(
    raw: pd.DataFrame,
    prior: float,
    snapshots: SeasonSnapshots,
    *,
    extended: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct the compact base or extended hierarchical representation."""
    rows = raw.reset_index(drop=True)
    output = _row_features(rows, prior)
    output = _attach_current_season(
        output, rows, snapshots.pitcher, "pitcher_id", "asof_pitcher_n",
        PITCHER_RATES, "season_pitcher", prior,
    )

    recent_columns = [
        f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)
    ]
    recent = rows[recent_columns].apply(pd.to_numeric, errors="coerce")
    recent_mean = recent.mean(axis=1).fillna(prior)
    recent_std = recent.std(axis=1).fillna(0.15).clip(0.0, 0.5)
    career_n = pd.to_numeric(
        rows["asof_pitcher_n"], errors="coerce",
    ).fillna(0).clip(lower=0)
    career_rate = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce",
    ).fillna(prior)
    strength = (
        55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(career_n))
    ).clip(50.0, 180.0)
    career_base = (career_rate * career_n + prior * strength) / (
        career_n + strength
    )
    season_n = output["season_pitcher_n"].clip(lower=0)
    season_raw = output["season_pitcher_success_rate_raw"].fillna(prior)
    season_estimate = (season_raw * season_n + prior * 30.0) / (season_n + 30.0)
    season_reliability = season_n / (season_n + 80.0)
    hierarchy = career_base + (
        0.15 + 0.30 * season_reliability
    ) * (season_estimate - career_base)

    output["recent_success_std"] = recent_std.to_numpy(np.float32)
    output["dynamic_smoothing_strength"] = strength.to_numpy(np.float32)
    output["career_dynamic_base"] = career_base.to_numpy(np.float32)
    output["season_form_estimate"] = season_estimate.to_numpy(np.float32)
    output["season_form_reliability"] = season_reliability.to_numpy(np.float32)
    output["hierarchical_success_base"] = hierarchy.to_numpy(np.float32)
    output["season_recent_gap"] = (season_estimate - recent_mean).to_numpy(np.float32)
    output["stable_momentum"] = (
        (recent_mean - career_base) / (1.0 + 5.0 * recent_std)
    ).to_numpy(np.float32)

    pressure = pd.cut(
        pd.to_numeric(rows["li"], errors="coerce").fillna(0),
        [-np.inf, 0.75, 1.5, 3.0, np.inf],
        labels=["low", "normal", "high", "extreme"],
    ).astype(str)
    inning = pd.cut(
        pd.to_numeric(rows["inning"], errors="coerce").fillna(0),
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype(str)
    score = pd.cut(
        pd.to_numeric(rows["score_diff_pitcher_team"], errors="coerce").fillna(0),
        [-np.inf, -3, -1, 1, 3, np.inf],
        labels=["far_behind", "behind", "close", "ahead", "far_ahead"],
    ).astype(str)
    output["count_base_context"] = (
        output["count_state"].astype(str) + "|" + rows["base_state"].astype(str)
    )
    output["hand_count_context"] = (
        output["hand_matchup"].astype(str) + "|" + output["count_state"].astype(str)
    )
    output["inning_score_context"] = inning + "|" + score
    output["pressure_context"] = (
        pressure + "|" + rows["num_runners_on"].astype(str) + "|"
        + output["count_state"].astype(str)
    )
    output["recent_regime"] = np.select(
        [recent_std < 0.04, recent_std < 0.10],
        ["stable", "normal"], default="volatile",
    )

    if extended:
        output = _attach_current_season(
            output, rows, snapshots.batter, "batter_id", "asof_batter_n",
            BATTER_RATES, "season_batter", prior,
        )
        output = _attach_current_season(
            output, rows, snapshots.pitchmix, "pitcher_id",
            "asof_pitcher_pitchmix_n", PITCHMIX_RATES, "season_pitchmix",
            prior,
        )
        batter_n = output["season_batter_n"]
        batter_rate = output["season_batter_success_rate_raw"]
        output["season_batter_success_smoothed"] = (
            batter_rate * batter_n + prior * 40.0
        ) / (batter_n + 40.0)
        output["season_batter_reliability"] = batter_n / (batter_n + 80.0)
        mix_n = output["season_pitchmix_n"]
        smoothed_mix = []
        for short_name, career_column in (
            ("fastball_rate", "asof_pitcher_fastball_rate"),
            ("breaking_rate", "asof_pitcher_breaking_rate"),
            ("offspeed_rate", "asof_pitcher_offspeed_rate"),
        ):
            raw_column = f"season_pitchmix_{short_name}_raw"
            column = f"season_pitchmix_{short_name}_smoothed"
            career = pd.to_numeric(
                rows[career_column], errors="coerce",
            ).fillna(0.0)
            output[column] = (
                output[raw_column] * mix_n + career * 50.0
            ) / (mix_n + 50.0)
            smoothed_mix.append(column)
        safe_mix = output[smoothed_mix].clip(1e-7, 1.0)
        output["season_pitchmix_entropy"] = -(
            safe_mix * np.log(safe_mix)
        ).sum(axis=1)
        output["season_pitchmix_reliability"] = mix_n / (mix_n + 100.0)
        output["pitcher_batter_form_gap"] = (
            output["hierarchical_success_base"]
            - output["season_batter_success_smoothed"]
        )

    for column in BASE_CATEGORICAL:
        output[column] = (
            output[column].astype("string").fillna("__MISSING__").astype(str)
        )
    numeric = [column for column in output if column not in BASE_CATEGORICAL]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan)
    return output, hierarchy.to_numpy(np.float32)


def snapshot_payload(snapshots: SeasonSnapshots) -> dict[str, dict[str, list]]:
    """Serialize compact frozen tables into JSON-safe column-oriented records."""
    payload: dict[str, dict[str, list]] = {}
    for name in ("pitcher", "batter", "pitchmix"):
        table = getattr(snapshots, name)
        payload[name] = {
            column: table[column].where(table[column].notna(), None).tolist()
            for column in table.columns
        }
    return payload


def snapshots_from_payload(payload: dict[str, dict[str, list]]) -> SeasonSnapshots:
    tables = {name: pd.DataFrame(columns) for name, columns in payload.items()}
    return SeasonSnapshots(
        pitcher=tables["pitcher"], batter=tables["batter"],
        pitchmix=tables["pitchmix"],
    )


def build_anchor_residual_features(
    rows: pd.DataFrame,
    anchor: np.ndarray,
    hierarchical_prediction: np.ndarray,
) -> pd.DataFrame:
    """Small, row-local feature set for correcting a strong anchor's residual."""
    anchor = np.asarray(anchor, dtype=float)
    ours = np.asarray(hierarchical_prediction, dtype=float)
    if len(rows) != len(anchor) or len(rows) != len(ours):
        raise ValueError("residual feature inputs have different lengths")
    gap = ours - anchor
    output = pd.DataFrame({
        "anchor": anchor,
        "hierarchical_prediction": ours,
        "prediction_gap": gap,
        "absolute_prediction_gap": np.abs(gap),
        "squared_prediction_gap": np.square(gap),
        "prediction_midpoint": 0.5 * (anchor + ours),
        "anchor_uncertainty": anchor * (1.0 - anchor),
    })
    numeric_columns = (
        "game_month", "game_dayofweek", "inning", "balls_before",
        "strikes_before", "outs_before", "score_diff_home",
        "score_diff_pitcher_team", "num_runners_on", "home_win_expectancy",
        "li", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(rows[column], errors="coerce").to_numpy()
    output["log_pitcher_n"] = np.log1p(output["asof_pitcher_n"].clip(lower=0))
    output["log_batter_n"] = np.log1p(output["asof_batter_n"].clip(lower=0))
    output["pitcher_batter_career_gap"] = (
        output["asof_pitcher_success_rate"]
        - output["asof_batter_success_rate"]
    )
    output["recent_success_mean"] = output[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].mean(axis=1)
    output["recent_success_std"] = output[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].std(axis=1)
    output["count_state"] = (
        rows["balls_before"].astype(str) + "-"
        + rows["strikes_before"].astype(str)
    )
    output["hand_matchup"] = (
        rows["pitcher_hand"].astype(str) + "-"
        + rows["batter_hand"].astype(str)
    )
    for column in ("top_bottom", "base_state", "pitcher_hand", "batter_hand"):
        output[column] = rows[column].astype("string").fillna("__MISSING__").astype(str)
    for column in ("count_state", "hand_matchup"):
        output[column] = output[column].astype("string").fillna("__MISSING__").astype(str)
    numeric = [column for column in output if column not in RESIDUAL_CATEGORICAL]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan)
    return output
