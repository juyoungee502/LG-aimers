"""Leakage-safe Trackman repertoire context features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PITCH_GROUPS = ("breaking", "fastball", "offspeed")
FEATURE_COLUMNS = tuple(
    [f"trackman_count_{name}_deviation" for name in PITCH_GROUPS]
    + ["trackman_count_speed_deviation"]
    + [f"trackman_hand_{name}_deviation" for name in PITCH_GROUPS]
    + ["trackman_hand_speed_deviation"]
)


def pitcher_mapping(root: Path, main: pd.DataFrame, trackman: pd.DataFrame):
    """Vote anonymous-to-Trackman pitcher IDs from reliable pitch alignments."""
    with np.load(root / "outputs" / "trackman_pitch_alignment.npz") as aligned:
        links = pd.DataFrame({
            "row_id": aligned["row_id"].astype(str),
            "trackman_id": aligned["trackman_id"].astype(str),
        })
    main_ids = main[["row_id", "pitcher_id"]].copy()
    main_ids["row_id"] = main_ids["row_id"].astype(str)
    track_ids = trackman[["trackman_id", "pitcher_trackman_id"]].copy()
    track_ids["trackman_id"] = track_ids["trackman_id"].astype(str)
    pairs = links.merge(main_ids, on="row_id", how="left", validate="one_to_one")
    pairs = pairs.merge(track_ids, on="trackman_id", how="left", validate="one_to_one")
    votes = pairs.groupby(
        ["pitcher_id", "pitcher_trackman_id"], observed=True,
    ).size().rename("votes").reset_index()
    totals = votes.groupby("pitcher_id")["votes"].transform("sum")
    votes["confidence"] = votes["votes"] / totals
    best = votes.sort_values(
        ["pitcher_id", "votes"], ascending=[True, False],
    ).drop_duplicates("pitcher_id")
    best = best.loc[(best["votes"] >= 3) & (best["confidence"] >= .90)]
    return {
        int(trackman_id): int(pitcher_id)
        for pitcher_id, trackman_id in best[
            ["pitcher_id", "pitcher_trackman_id"]
        ].itertuples(index=False, name=None)
    }, best


def prepare_trackman(trackman: pd.DataFrame, mapping: dict[int, int]):
    frame = trackman.copy()
    frame["pitcher_id"] = frame["pitcher_trackman_id"].map(mapping)
    frame = frame.dropna(subset=["pitcher_id"]).copy()
    frame["pitcher_id"] = frame["pitcher_id"].astype(np.int64)
    for group in PITCH_GROUPS:
        frame[f"is_{group}"] = frame["pitch_type_group"].eq(group).astype(np.float32)
    return frame


def summarize(frame, keys, prefix, minimum):
    grouped = frame.groupby(keys, observed=True, sort=False)
    output = pd.DataFrame({f"{prefix}_n": grouped.size()})
    for group in PITCH_GROUPS:
        output[f"{prefix}_{group}"] = grouped[f"is_{group}"].mean()
    output[f"{prefix}_speed"] = grouped["rel_speed"].mean()
    return output.loc[output[f"{prefix}_n"] >= minimum]


def deviation_table(table, baseline, prefix):
    output = table.join(baseline, on="pitcher_id")
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
        return None, None
    baseline = summarize(past, ["pitcher_id"], "base", 0)
    count = deviation_table(
        summarize(
            past, ["pitcher_id", "balls_before", "strikes_before"],
            "trackman_count", 30,
        ), baseline, "trackman_count",
    )
    hand = deviation_table(
        summarize(
            past, ["pitcher_id", "batter_hand"], "trackman_hand", 50,
        ), baseline, "trackman_hand",
    )
    return count, hand


def attach_context(rows: pd.DataFrame, trackman: pd.DataFrame):
    """Attach features using only Trackman seasons before each row's season."""
    blocks = []
    hand_codes = {1: "Left", 2: "Right"}
    for season, part in rows.groupby("season", sort=False):
        count, hand = context_tables(trackman, int(season))
        output = pd.DataFrame(index=part.index, columns=FEATURE_COLUMNS, dtype=np.float32)
        if count is None:
            blocks.append(output)
            continue
        left = part[["pitcher_id", "balls_before", "strikes_before", "batter_hand"]].copy()
        left["_order"] = np.arange(len(left))
        left["batter_hand_name"] = left["batter_hand"].map(hand_codes)
        count_frame = count.reset_index()
        got_count = left.merge(
            count_frame, on=["pitcher_id", "balls_before", "strikes_before"],
            how="left", sort=False,
        ).sort_values("_order")
        hand_frame = hand.reset_index().rename(columns={"batter_hand": "batter_hand_name"})
        got_hand = left.merge(
            hand_frame, on=["pitcher_id", "batter_hand_name"],
            how="left", sort=False,
        ).sort_values("_order")
        count_columns = [column for column in FEATURE_COLUMNS if "_count_" in column]
        hand_columns = [column for column in FEATURE_COLUMNS if "_hand_" in column]
        output[count_columns] = got_count[count_columns].to_numpy(np.float32)
        output[hand_columns] = got_hand[hand_columns].to_numpy(np.float32)
        blocks.append(output)
    return pd.concat(blocks).sort_index().reindex(columns=FEATURE_COLUMNS)


def freeze_context(trackman: pd.DataFrame, target_season: int):
    """Serialize final lookup tables so inference needs no Trackman file."""
    count, hand = context_tables(trackman, target_season)
    if count is None:
        raise ValueError(f"No Trackman history before {target_season}")
    count_frame = count.reset_index()
    hand_frame = hand.reset_index()
    count_columns = list(count.columns)
    hand_columns = list(hand.columns)
    return {
        "target_season": int(target_season),
        "feature_columns": list(FEATURE_COLUMNS),
        "count_columns": count_columns,
        "count_keys": [
            f"{int(pitcher)}:{int(balls)}:{int(strikes)}"
            for pitcher, balls, strikes in count_frame[
                ["pitcher_id", "balls_before", "strikes_before"]
            ].itertuples(index=False, name=None)
        ],
        "count_values": count_frame[count_columns].astype(np.float32).values.tolist(),
        "hand_columns": hand_columns,
        "hand_keys": [
            f"{int(pitcher)}:{handedness}"
            for pitcher, handedness in hand_frame[
                ["pitcher_id", "batter_hand"]
            ].itertuples(index=False, name=None)
        ],
        "hand_values": hand_frame[hand_columns].astype(np.float32).values.tolist(),
    }


def apply_frozen_context(rows: pd.DataFrame, configuration: dict):
    """Apply frozen row-local Trackman context lookups."""
    output = pd.DataFrame(
        np.nan, index=rows.index, columns=configuration["feature_columns"],
        dtype=np.float32,
    )
    count_lookup = dict(zip(
        configuration["count_keys"], configuration["count_values"]
    ))
    count_keys = (
        rows["pitcher_id"].astype(str) + ":"
        + rows["balls_before"].astype(str) + ":"
        + rows["strikes_before"].astype(str)
    )
    count_values = count_keys.map(count_lookup)
    count_matrix = np.full(
        (len(rows), len(configuration["count_columns"])), np.nan, np.float32,
    )
    for index, value in enumerate(count_values):
        if isinstance(value, list):
            count_matrix[index] = value
    output[configuration["count_columns"]] = count_matrix

    hand_lookup = dict(zip(
        configuration["hand_keys"], configuration["hand_values"]
    ))
    handedness = rows["batter_hand"].map({1: "Left", 2: "Right"})
    hand_keys = rows["pitcher_id"].astype(str) + ":" + handedness.astype(str)
    hand_values = hand_keys.map(hand_lookup)
    hand_matrix = np.full(
        (len(rows), len(configuration["hand_columns"])), np.nan, np.float32,
    )
    for index, value in enumerate(hand_values):
        if isinstance(value, list):
            hand_matrix[index] = value
    output[configuration["hand_columns"]] = hand_matrix
    return output.reindex(columns=configuration["feature_columns"])
