"""Row-independent inference for the v66 nested context deviations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _keys(rows: pd.DataFrame) -> dict[str, pd.Series]:
    pitcher = rows["pitcher_id"].astype(str)
    hand = rows["batter_hand"].astype(str)
    pitcher_hand = pitcher.str.cat(hand, sep="|")
    advantage = rows["strikes_before"].gt(rows["balls_before"]).astype(np.int8)
    on_base = rows["num_runners_on"].gt(0).astype(np.int8)
    return {
        "platoon": pitcher_hand,
        "advantage": pitcher_hand.str.cat(advantage.astype(str), sep="|"),
        "runner": pitcher_hand.str.cat(on_base.astype(str), sep="|"),
    }


def apply_nested_deviations(
    rows: pd.DataFrame, configuration: dict,
) -> np.ndarray:
    """Look up frozen train-only contrasts using each evaluation row alone."""
    lookup_keys = _keys(rows)
    correction = np.zeros(len(rows), dtype=np.float64)
    for axis in configuration["axes"]:
        name = str(axis["name"])
        table = dict(zip(axis["keys"], axis["deltas"]))
        values = lookup_keys[name].map(table).fillna(
            float(axis.get("unknown_key_delta", 0.0))
        ).to_numpy(np.float64)
        correction += float(axis["weight"]) * values
    return correction
