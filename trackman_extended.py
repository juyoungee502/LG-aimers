"""Prior-season Trackman pitcher profiles for diagnostic specialists."""
from __future__ import annotations

import numpy as np
import pandas as pd


PITCH_GROUPS = ("breaking", "fastball", "offspeed")
METRICS = (
    "rel_speed", "spin_rate", "induced_vert_break", "arm_horz_break",
    "extension", "rel_height", "arm_rel_side", "zone_speed",
)
GROUP_METRICS = (
    "rel_speed", "spin_rate", "induced_vert_break", "arm_horz_break",
)


def prepare_physical(trackman: pd.DataFrame, mapping: dict[int, int]):
    frame = trackman.copy()
    frame["pitcher_id"] = frame["pitcher_trackman_id"].map(mapping)
    frame = frame.dropna(subset=["pitcher_id"]).copy()
    frame["pitcher_id"] = frame["pitcher_id"].astype(np.int64)
    direction = np.where(frame["pitcher_hand"].eq("Left"), -1., 1.)
    frame["arm_horz_break"] = frame["horz_break"] * direction
    frame["arm_rel_side"] = frame["rel_side"] * direction
    for group in PITCH_GROUPS:
        frame[f"is_{group}"] = frame["pitch_type_group"].eq(group).astype(np.float32)
    return frame


def _profile(frame: pd.DataFrame, prefix: str):
    grouped = frame.groupby("pitcher_id", observed=True, sort=False)
    output = pd.DataFrame({f"{prefix}_n": grouped.size()})
    for metric in METRICS:
        output[f"{prefix}_{metric}_mean"] = grouped[metric].mean()
        output[f"{prefix}_{metric}_std"] = grouped[metric].std()
    for group in PITCH_GROUPS:
        output[f"{prefix}_{group}_rate"] = grouped[f"is_{group}"].mean()
    return output


def _group_profile(frame: pd.DataFrame):
    useful = frame.loc[frame["pitch_type_group"].isin(PITCH_GROUPS)]
    grouped = useful.groupby(
        ["pitcher_id", "pitch_type_group"], observed=True, sort=False,
    )
    pieces = [grouped.size().unstack("pitch_type_group").add_prefix("tm_group_n_")]
    for metric in GROUP_METRICS:
        table = grouped[metric].mean().unstack("pitch_type_group")
        pieces.append(table.add_prefix(f"tm_group_{metric}_"))
    return pd.concat(pieces, axis=1)


def profile_table(trackman: pd.DataFrame, target_season: int):
    past = trackman.loc[trackman["season"].lt(target_season)]
    if past.empty:
        return None
    career = _profile(past, "tm_career")
    recent = _profile(past.loc[past["season"].eq(target_season - 1)], "tm_recent")
    output = career.join(recent, how="left")
    for metric in METRICS:
        for statistic in ("mean", "std"):
            output[f"tm_delta_{metric}_{statistic}"] = (
                output[f"tm_recent_{metric}_{statistic}"]
                - output[f"tm_career_{metric}_{statistic}"]
            )
    for group in PITCH_GROUPS:
        output[f"tm_delta_{group}_rate"] = (
            output[f"tm_recent_{group}_rate"] - output[f"tm_career_{group}_rate"]
        )
    output["tm_career_log_n"] = np.log1p(output["tm_career_n"])
    output["tm_recent_log_n"] = np.log1p(output["tm_recent_n"])
    return output.join(_group_profile(past), how="left")


def attach_profiles(rows: pd.DataFrame, trackman: pd.DataFrame):
    blocks = []
    columns = None
    for season, part in rows.groupby("season", sort=False):
        table = profile_table(trackman, int(season))
        if table is None:
            continue
        if columns is None:
            columns = list(table.columns)
        block = part[["pitcher_id"]].join(table, on="pitcher_id")[columns]
        blocks.append(block)
    if not blocks:
        return pd.DataFrame(index=rows.index)
    output = pd.concat(blocks).sort_index()
    return output.reindex(index=rows.index, columns=columns)
