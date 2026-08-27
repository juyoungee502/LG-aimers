"""Frozen, row-independent features and corrections for the v24 candidate."""
from __future__ import annotations

import numpy as np
import pandas as pd


COMMAND_CONTEXT_SPECS = (
    ("hand", ("pitcher_id", "batter_hand"), 200.0),
    ("count", ("pitcher_id", "count_state"), 300.0),
    ("hand_count", ("pitcher_id", "batter_hand", "count_state"), 500.0),
)
RESOLUTION_CONTEXTS = {
    "regime_count": ("game_type", "count_state"),
    "regime_count_hands": (
        "game_type", "count_state", "pitcher_hand", "batter_hand",
    ),
    "regime_count_runners": ("game_type", "count_state", "runner_gate"),
}
POLICY = {
    "command_no_month": 1.70,
    "command_full": 0.15,
    "command_recent": 1.20,
    "global_logit_shift": -0.002,
    "f_count": 0.55,
    "f_hands": -0.25,
    "f_runners": -0.25,
    "pressure_hand": 0.25,
}
COMMAND_BLEND_SCALE = 0.35
PRESSURE_SHRINK = 1200.0
EARLY_PITCHER_PITCHES = 600.0


def _token(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _keys(frame: pd.DataFrame, columns) -> pd.Series:
    columns = list(columns)
    result = frame[columns[0]].map(_token)
    for column in columns[1:]:
        result = result + "|" + frame[column].map(_token)
    return result


def _command_source_table(source: pd.DataFrame):
    global_rate = float(source["_target"].mean()) if len(source) else 0.5
    pitcher = source.groupby("pitcher_id", observed=True)["_target"].agg(
        pitcher_sum="sum", pitcher_n="count",
    ).reset_index()
    pitcher["pitcher_rate"] = (
        pitcher["pitcher_sum"] + 500.0 * global_rate
    ) / (pitcher["pitcher_n"] + 500.0)
    return global_rate, pitcher


def _command_part(
    query: pd.DataFrame, source: pd.DataFrame, order: np.ndarray,
) -> pd.DataFrame:
    global_rate, pitcher = _command_source_table(source)
    result = query.copy()
    result["_order"] = order
    result = result.merge(
        pitcher[["pitcher_id", "pitcher_rate"]],
        on="pitcher_id", how="left", sort=False,
    )
    result["prior_command_pitcher_rate"] = result["pitcher_rate"].fillna(
        global_rate,
    )
    for name, keys, shrink in COMMAND_CONTEXT_SPECS:
        table = source.groupby(list(keys), observed=True)["_target"].agg(
            context_sum="sum", context_n="count",
        ).reset_index()
        table = table.merge(
            pitcher[["pitcher_id", "pitcher_rate"]],
            on="pitcher_id", how="left", sort=False,
        )
        rate = f"prior_command_{name}_rate"
        weight = f"prior_command_{name}_weight"
        delta = f"prior_command_{name}_delta"
        table[rate] = (
            table["context_sum"]
            + shrink * table["pitcher_rate"].fillna(global_rate)
        ) / (table["context_n"] + shrink)
        table[weight] = table["context_n"] / (table["context_n"] + shrink)
        result = result.merge(
            table[[*keys, rate, weight]],
            on=list(keys), how="left", sort=False,
        )
        result[rate] = result[rate].fillna(result["prior_command_pitcher_rate"])
        result[weight] = result[weight].fillna(0.0)
        result[delta] = result[rate] - result["prior_command_pitcher_rate"]
    columns = [
        "_order", "prior_command_pitcher_rate",
        *[
            f"prior_command_{name}_{suffix}"
            for name, _, _ in COMMAND_CONTEXT_SPECS
            for suffix in ("rate", "weight", "delta")
        ],
    ]
    return result[columns]


def time_safe_command_features(
    rows: pd.DataFrame, target: np.ndarray, history_window: int | None = None,
) -> pd.DataFrame:
    """Create training features using only seasons before each source row."""
    work = rows[[
        "season", "pitcher_id", "batter_hand", "balls_before", "strikes_before",
    ]].copy()
    work["count_state"] = (
        work["balls_before"].to_numpy(np.int16) * 3
        + work["strikes_before"].to_numpy(np.int16)
    )
    work["_target"] = np.asarray(target, dtype=float)
    parts = []
    for season in np.sort(work["season"].unique()):
        active = work["season"].eq(season)
        source_mask = work["season"].lt(season)
        if history_window is not None:
            source_mask &= work["season"].ge(season - history_window)
        query = work.loc[active].drop(columns="_target")
        parts.append(_command_part(query, work.loc[source_mask], query.index.to_numpy()))
    result = pd.concat(parts).sort_values("_order").drop(columns="_order")
    result.index = rows.index
    return result.astype(np.float32)


def freeze_command(
    rows: pd.DataFrame, target: np.ndarray, target_season: int,
    history_window: int | None = None,
) -> dict:
    work = rows[[
        "season", "pitcher_id", "batter_hand", "balls_before", "strikes_before",
    ]].copy()
    work["count_state"] = (
        work["balls_before"].to_numpy(np.int16) * 3
        + work["strikes_before"].to_numpy(np.int16)
    )
    work["_target"] = np.asarray(target, dtype=float)
    source = work.loc[work["season"].lt(target_season)]
    if history_window is not None:
        source = source.loc[source["season"].ge(target_season - history_window)]
    global_rate, pitcher = _command_source_table(source)
    configuration = {
        "global_rate": global_rate,
        "history_window": history_window,
        "pitcher": dict(zip(
            pitcher["pitcher_id"].map(_token), pitcher["pitcher_rate"].astype(float),
        )),
        "contexts": {},
    }
    for name, keys, shrink in COMMAND_CONTEXT_SPECS:
        table = source.groupby(list(keys), observed=True)["_target"].agg(
            context_sum="sum", context_n="count",
        ).reset_index()
        table = table.merge(
            pitcher[["pitcher_id", "pitcher_rate"]],
            on="pitcher_id", how="left", sort=False,
        )
        rate = (
            table["context_sum"] + shrink * table["pitcher_rate"].fillna(global_rate)
        ) / (table["context_n"] + shrink)
        weight = table["context_n"] / (table["context_n"] + shrink)
        configuration["contexts"][name] = {
            "keys": list(keys),
            "values": dict(zip(
                _keys(table, keys),
                [[float(r), float(w)] for r, w in zip(rate, weight)],
            )),
        }
    return configuration


def apply_frozen_command(rows: pd.DataFrame, configuration: dict) -> pd.DataFrame:
    count_state = (
        rows["balls_before"].to_numpy(np.int16) * 3
        + rows["strikes_before"].to_numpy(np.int16)
    )
    query = rows.copy()
    query["count_state"] = count_state
    global_rate = float(configuration["global_rate"])
    pitcher_rate = query["pitcher_id"].map(_token).map(
        configuration["pitcher"],
    ).fillna(global_rate).to_numpy(float)
    result = pd.DataFrame({
        "prior_command_pitcher_rate": pitcher_rate,
    }, index=rows.index)
    for name, _, _ in COMMAND_CONTEXT_SPECS:
        spec = configuration["contexts"][name]
        pairs = _keys(query, spec["keys"]).map(spec["values"])
        rate = np.asarray([
            value[0] if isinstance(value, list) else pitcher_rate[index]
            for index, value in enumerate(pairs.tolist())
        ], dtype=np.float32)
        weight = np.asarray([
            value[1] if isinstance(value, list) else 0.0
            for value in pairs.tolist()
        ], dtype=np.float32)
        result[f"prior_command_{name}_rate"] = rate
        result[f"prior_command_{name}_weight"] = weight
        result[f"prior_command_{name}_delta"] = rate - pitcher_rate
    return result.astype(np.float32)


def pressure_state(rows: pd.DataFrame) -> np.ndarray:
    balls = rows["balls_before"].to_numpy(np.int16)
    strikes = rows["strikes_before"].to_numpy(np.int16)
    return np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)


def freeze_pressure(rows: pd.DataFrame, target: np.ndarray) -> dict:
    source = rows[[
        "season", "game_type", "pitcher_id", "batter_hand",
        "balls_before", "strikes_before",
    ]].copy()
    source["pressure_state"] = pressure_state(source)
    source["_target"] = np.asarray(target, dtype=float)
    source["_relative"] = source["_target"] - source.groupby(
        ["season", "game_type"], observed=True,
    )["_target"].transform("mean")
    pitcher = source.groupby("pitcher_id", observed=True)["_relative"].agg(
        parent_sum="sum", parent_n="count",
    ).reset_index()
    pitcher["parent_rate"] = pitcher["parent_sum"] / pitcher["parent_n"]
    keys = ("pitcher_id", "batter_hand", "pressure_state")
    child = source.groupby(list(keys), observed=True)["_relative"].agg(
        child_sum="sum", child_n="count",
    ).reset_index()
    child = child.merge(
        pitcher[["pitcher_id", "parent_rate"]],
        on="pitcher_id", how="left", validate="many_to_one",
    )
    child_rate = child["child_sum"] / child["child_n"]
    child["deviation"] = (
        child["child_n"] / (child["child_n"] + PRESSURE_SHRINK)
        * (child_rate - child["parent_rate"])
    )
    return {
        "keys": list(keys), "shrink": PRESSURE_SHRINK,
        "values": dict(zip(_keys(child, keys), child["deviation"].astype(float))),
    }


def apply_frozen_pressure(rows: pd.DataFrame, configuration: dict) -> np.ndarray:
    query = rows.copy()
    query["pressure_state"] = pressure_state(query)
    return _keys(query, configuration["keys"]).map(
        configuration["values"],
    ).fillna(0.0).to_numpy(float)


def freeze_pitcher_season_origins(rows: pd.DataFrame) -> dict:
    latest_season = int(rows["season"].max())
    latest = rows.loc[rows["season"].eq(latest_season)]
    last = latest.groupby("pitcher_id", observed=True, sort=False).tail(1)
    return {
        _token(pitcher): float(max(0.0, count + 1.0))
        for pitcher, count in zip(
            last["pitcher_id"], last["asof_pitcher_n"].fillna(0.0),
        )
    }


def early_pitcher_gate(rows: pd.DataFrame, origins: dict) -> np.ndarray:
    origin = rows["pitcher_id"].map(_token).map(origins).fillna(0.0).to_numpy(float)
    current = rows["asof_pitcher_n"].fillna(0.0).to_numpy(float)
    return np.maximum(0.0, current - origin) <= EARLY_PITCHER_PITCHES


def resolution_context_frame(rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    result["season"] = rows["season"].to_numpy()
    result["game_type"] = rows["game_type"].astype(str).to_numpy()
    result["count_state"] = (
        rows["balls_before"].to_numpy(np.int16) * 3
        + rows["strikes_before"].to_numpy(np.int16)
    )
    result["pitcher_hand"] = rows["pitcher_hand"].to_numpy()
    result["batter_hand"] = rows["batter_hand"].to_numpy()
    result["runner_gate"] = rows["num_runners_on"].gt(0).to_numpy(np.int8)
    return result


def resolution_label(context: pd.DataFrame, target: np.ndarray, mode: str) -> np.ndarray:
    keys = RESOLUTION_CONTEXTS[mode]
    frame = context[["season", *keys]].copy()
    frame["target"] = np.asarray(target, dtype=float)
    center = frame.groupby(
        ["season", *keys], observed=True,
    )["target"].transform("mean").to_numpy(float)
    return np.asarray(target, dtype=float) - center


def freeze_resolution_center(
    context: pd.DataFrame, prediction: np.ndarray, mode: str,
) -> dict:
    keys = RESOLUTION_CONTEXTS[mode]
    source = context[list(keys)].copy()
    source["prediction"] = np.asarray(prediction, dtype=float)
    table = source.groupby(list(keys), observed=True)["prediction"].mean().reset_index()
    fallback = source.groupby("game_type", observed=True)["prediction"].mean()
    return {
        "keys": list(keys),
        "values": dict(zip(_keys(table, keys), table["prediction"].astype(float))),
        "game_type_fallback": {str(key): float(value) for key, value in fallback.items()},
        "global_fallback": float(source["prediction"].mean()),
    }


def apply_resolution_center(
    rows: pd.DataFrame, prediction: np.ndarray, configuration: dict,
) -> np.ndarray:
    context = resolution_context_frame(rows)
    center = _keys(context, configuration["keys"]).map(
        configuration["values"],
    )
    missing = center.isna()
    if missing.any():
        center.loc[missing] = context.loc[missing, "game_type"].map(
            configuration["game_type_fallback"],
        )
    center = center.fillna(float(configuration["global_fallback"])).to_numpy(float)
    return np.asarray(prediction, dtype=float) - center
