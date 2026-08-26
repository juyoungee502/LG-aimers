"""Leakage-safe prior-season context features for failure specialists."""
from __future__ import annotations

import numpy as np
import pandas as pd


CONTEXT_SPECS = {
    "pitcher_count": ["pitcher_id", "count_state"],
    "pitcher_hand": ["pitcher_id", "batter_hand"],
    "pitcher_game": ["pitcher_id", "game_type"],
    "batter_count": ["batter_id", "count_state"],
    "batter_hand": ["batter_id", "pitcher_hand"],
}
DETAIL_LABELS = ("reverse", "middle", "ball", "strike")


def prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[[
        "season", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "game_type",
    ]].copy()
    out["count_state"] = (
        frame["balls_before"] * 3 + frame["strikes_before"]
    ).astype(np.int8)
    return out


def feature_columns() -> list[str]:
    return [
        f"failure_context_{context}_{label}"
        for context in CONTEXT_SPECS
        for label in ("success", *DETAIL_LABELS)
    ]


def prior_season_context(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    rows: pd.DataFrame | None = None,
    smoothing: float = 200.0,
) -> pd.DataFrame:
    """Build row-local features using only the immediately preceding season."""
    if rows is None:
        rows = history
    source = prepare_keys(history)
    source["success"] = history["control_success"].to_numpy(np.float32)
    for label in DETAIL_LABELS:
        source[label] = labels[label].to_numpy(np.float32)
    row_keys = prepare_keys(rows)
    blocks = []
    columns = feature_columns()
    for season, part in row_keys.groupby("season", sort=False):
        prior = source.loc[source["season"].eq(int(season) - 1)]
        block = pd.DataFrame(index=part.index, columns=columns, dtype=np.float32)
        if prior.empty:
            blocks.append(block)
            continue
        for context, keys in CONTEXT_SPECS.items():
            left = part[keys].copy()
            left["_order"] = np.arange(len(left))
            for label in ("success", *DETAIL_LABELS):
                league = float(prior[label].mean())
                grouped = prior.groupby(keys, observed=True, sort=False)[label].agg(
                    ["sum", "count"]
                ).reset_index()
                value = f"failure_context_{context}_{label}"
                grouped[value] = (
                    (grouped["sum"] + smoothing * league)
                    / (grouped["count"] + smoothing) - league
                )
                mapped = left.merge(
                    grouped[[*keys, value]], on=keys, how="left", sort=False,
                ).sort_values("_order")
                block[value] = mapped[value].to_numpy(np.float32)
        blocks.append(block)
    return pd.concat(blocks).sort_index().reindex(columns=columns)
