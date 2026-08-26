"""Leakage-safe Trackman batter and pitcher-batter context features.

Every lookup is built from seasons strictly before the row being transformed.
Only deviations are exposed to the model: raw sample counts are deliberately
omitted because they encode changing Trackman coverage across seasons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PITCH_GROUPS = ("breaking", "fastball", "offspeed")
CONTEXTS = ("tm_joint", "tm_batter_count", "tm_batter_phand", "tm_matchup")
FEATURE_COLUMNS = tuple(
    column
    for prefix in CONTEXTS
    for column in (
        *(f"{prefix}_{group}_deviation" for group in PITCH_GROUPS),
        f"{prefix}_speed_deviation",
    )
)


def _identity_mapping(
    root: Path,
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    main_column: str,
    trackman_column: str,
):
    """Vote an anonymous-to-Trackman identity map from aligned pitches."""
    with np.load(root / "outputs" / "trackman_pitch_alignment.npz") as aligned:
        links = pd.DataFrame({
            "row_id": aligned["row_id"].astype(str),
            "trackman_id": aligned["trackman_id"].astype(str),
        })
    main_ids = main[["row_id", main_column]].copy()
    main_ids["row_id"] = main_ids["row_id"].astype(str)
    track_ids = trackman[["trackman_id", trackman_column]].copy()
    track_ids["trackman_id"] = track_ids["trackman_id"].astype(str)
    pairs = links.merge(main_ids, on="row_id", how="left", validate="one_to_one")
    pairs = pairs.merge(track_ids, on="trackman_id", how="left", validate="one_to_one")
    votes = pairs.groupby(
        [main_column, trackman_column], observed=True,
    ).size().rename("votes").reset_index()
    totals = votes.groupby(main_column)["votes"].transform("sum")
    votes["confidence"] = votes["votes"] / totals
    best = votes.sort_values(
        [main_column, "votes"], ascending=[True, False],
    ).drop_duplicates(main_column)
    best = best.loc[(best["votes"] >= 3) & (best["confidence"] >= .90)]
    mapping = {
        int(trackman_id): int(main_id)
        for main_id, trackman_id in best[
            [main_column, trackman_column]
        ].itertuples(index=False, name=None)
    }
    return mapping, best


def batter_mapping(root: Path, main: pd.DataFrame, trackman: pd.DataFrame):
    return _identity_mapping(
        root, main, trackman, "batter_id", "batter_trackman_id",
    )


def prepare_matchups(
    trackman: pd.DataFrame,
    pitcher_map: dict[int, int],
    batter_map: dict[int, int],
):
    frame = trackman.copy()
    frame["pitcher_id"] = frame["pitcher_trackman_id"].map(pitcher_map)
    frame["batter_id"] = frame["batter_trackman_id"].map(batter_map)
    frame = frame.dropna(subset=["pitcher_id", "batter_id"]).copy()
    frame[["pitcher_id", "batter_id"]] = frame[
        ["pitcher_id", "batter_id"]
    ].astype(np.int64)
    for group in PITCH_GROUPS:
        frame[f"is_{group}"] = frame["pitch_type_group"].eq(group).astype(np.float32)
    return frame


def _summarize(frame: pd.DataFrame, keys, prefix: str, minimum: int):
    grouped = frame.groupby(keys, observed=True, sort=False)
    output = pd.DataFrame({f"{prefix}_n": grouped.size()})
    for group in PITCH_GROUPS:
        output[f"{prefix}_{group}"] = grouped[f"is_{group}"].mean()
    output[f"{prefix}_speed"] = grouped["rel_speed"].mean()
    return output.loc[output[f"{prefix}_n"] >= minimum]


def _deviations(table, baseline, join_key: str, prefix: str):
    output = table.join(baseline, on=join_key)
    columns = []
    for group in PITCH_GROUPS:
        column = f"{prefix}_{group}_deviation"
        output[column] = output[f"{prefix}_{group}"] - output[f"base_{group}"]
        columns.append(column)
    speed_column = f"{prefix}_speed_deviation"
    output[speed_column] = output[f"{prefix}_speed"] - output["base_speed"]
    columns.append(speed_column)
    return output[columns]


def context_tables(trackman: pd.DataFrame, target_season: int):
    past = trackman.loc[trackman["season"].lt(target_season)]
    if past.empty:
        return {}
    pitcher_base = _summarize(past, ["pitcher_id"], "base", 0)
    batter_base = _summarize(past, ["batter_id"], "base", 0)
    return {
        "tm_joint": _deviations(
            _summarize(
                past,
                ["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
                "tm_joint", 20,
            ),
            pitcher_base, "pitcher_id", "tm_joint",
        ),
        "tm_batter_count": _deviations(
            _summarize(
                past, ["batter_id", "balls_before", "strikes_before"],
                "tm_batter_count", 20,
            ),
            batter_base, "batter_id", "tm_batter_count",
        ),
        "tm_batter_phand": _deviations(
            _summarize(
                past, ["batter_id", "pitcher_hand"], "tm_batter_phand", 30,
            ),
            batter_base, "batter_id", "tm_batter_phand",
        ),
        "tm_matchup": _deviations(
            _summarize(
                past, ["pitcher_id", "batter_id"], "tm_matchup", 20,
            ),
            pitcher_base, "pitcher_id", "tm_matchup",
        ),
    }


LOOKUP_KEYS = {
    "tm_joint": ["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
    "tm_batter_count": ["batter_id", "balls_before", "strikes_before"],
    "tm_batter_phand": ["batter_id", "pitcher_hand"],
    "tm_matchup": ["pitcher_id", "batter_id"],
}


def attach_matchup_context(rows: pd.DataFrame, trackman: pd.DataFrame):
    """Attach rolling prior-season matchup context to main-table rows."""
    blocks = []
    hand_codes = {1: "Left", 2: "Right"}
    for season, part in rows.groupby("season", sort=False):
        tables = context_tables(trackman, int(season))
        output = pd.DataFrame(
            np.nan, index=part.index, columns=FEATURE_COLUMNS, dtype=np.float32,
        )
        if not tables:
            blocks.append(output)
            continue
        left = part[[
            "pitcher_id", "batter_id", "balls_before", "strikes_before",
            "pitcher_hand", "batter_hand",
        ]].copy()
        left["pitcher_hand"] = left["pitcher_hand"].map(hand_codes)
        left["batter_hand"] = left["batter_hand"].map(hand_codes)
        for prefix, table in tables.items():
            columns = [column for column in FEATURE_COLUMNS if column.startswith(prefix + "_")]
            joined = left.join(table, on=LOOKUP_KEYS[prefix])
            output[columns] = joined[columns].to_numpy(np.float32)
        blocks.append(output)
    return pd.concat(blocks).sort_index().reindex(columns=FEATURE_COLUMNS)
