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


def _key_strings(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    values = frame[keys[0]].astype(str)
    for key in keys[1:]:
        values = values + ":" + frame[key].astype(str)
    return values


def freeze_prior_context(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    target_season: int,
    smoothing: float = 200.0,
) -> dict:
    """Serialize preceding-season context tables for row-local inference."""
    source = prepare_keys(history)
    source["success"] = history["control_success"].to_numpy(np.float32)
    for label in DETAIL_LABELS:
        source[label] = labels[label].to_numpy(np.float32)
    prior = source.loc[source["season"].eq(int(target_season) - 1)]
    if prior.empty:
        raise ValueError(f"No failure context for target season {target_season}")
    tables = {}
    value_labels = ("success", *DETAIL_LABELS)
    for context, keys in CONTEXT_SPECS.items():
        grouped = prior.groupby(keys, observed=True, sort=False)
        values = []
        for label in value_labels:
            league = float(prior[label].mean())
            sums = grouped[label].sum()
            counts = grouped[label].count()
            values.append(
                ((sums + smoothing * league) / (counts + smoothing) - league)
                .rename(f"failure_context_{context}_{label}")
            )
        table = pd.concat(values, axis=1).reset_index()
        tables[context] = {
            "keys": keys,
            "lookup_keys": _key_strings(table, keys).tolist(),
            "columns": [value.name for value in values],
            "values": table[[value.name for value in values]].astype(np.float32).values.tolist(),
        }
    return {
        "target_season": int(target_season), "smoothing": float(smoothing),
        "feature_columns": feature_columns(), "tables": tables,
    }


def apply_frozen_context(rows: pd.DataFrame, configuration: dict) -> pd.DataFrame:
    """Apply frozen failure-context tables without aggregating evaluation rows."""
    keys = prepare_keys(rows)
    output = pd.DataFrame(
        np.nan, index=rows.index, columns=configuration["feature_columns"],
        dtype=np.float32,
    )
    for context, table in configuration["tables"].items():
        lookup = dict(zip(table["lookup_keys"], table["values"]))
        mapped = _key_strings(keys, table["keys"]).map(lookup)
        matrix = np.full((len(rows), len(table["columns"])), np.nan, np.float32)
        for index, value in enumerate(mapped):
            if isinstance(value, list):
                matrix[index] = value
        output[table["columns"]] = matrix
    return output.reindex(columns=configuration["feature_columns"])
