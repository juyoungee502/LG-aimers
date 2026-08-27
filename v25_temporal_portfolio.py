"""Frozen row-independent residual tables for the v25 candidate.

The policies below were selected only after four chronological transfers and
61 segment constraints.  At inference time every lookup uses the current row
and tables learned from training data; evaluation rows are never aggregated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REGULAR_POLICY = (
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_success_rate", "bins": 16, "shrink": 6400., "scale": .5, "weight": .55},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_prev5_game_success_rate", "bins": 16, "shrink": 400., "scale": .25, "weight": .55},
    {"type": "one_d", "kind": "numeric", "column": "batter_team_id", "bins": 16, "shrink": 6400., "scale": .25, "weight": .10},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_prev3_game_middle_rate", "bins": 8, "shrink": 6400., "scale": .5, "weight": .15},
    {"type": "pair", "column": "pitcher_middle_season_s25", "context": "pitcher_hand", "bins": 4, "shrink": 400., "scale": .5, "weight": .10},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "num_runners_on", "bins": 4, "shrink": 400., "scale": .5, "weight": .10},
    {"type": "pair", "column": "pitcher_middle_season_rate", "context": "balls_before", "bins": 4, "shrink": 400., "scale": .5, "weight": .15},
    {"type": "pair", "column": "pitcher_season_minus_prior", "context": "inning_bucket", "bins": 8, "shrink": 1600., "scale": .25, "weight": .05},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "pitcher_hand", "bins": 8, "shrink": 400., "scale": .5, "weight": .40},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "pressure_state", "bins": 8, "shrink": 400., "scale": .5, "weight": .15},
    {"type": "pair", "column": "pitcher_middle_season_s100", "context": "batter_hand", "bins": 8, "shrink": 400., "scale": .25, "weight": .15},
)

FUTURES_POLICY = (
    {"type": "one_d", "kind": "numeric", "column": "pitcher_season_success_count", "bins": 16, "shrink": 25., "scale": .5, "weight": .20},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_reverse_rate", "bins": 8, "shrink": 25., "scale": .5, "weight": .05},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_season_success_s200", "bins": 4, "shrink": 100., "scale": .5, "weight": .50},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_reverse_delta_x_2strike", "bins": 8, "shrink": 400., "scale": .25, "weight": 1.},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_reverse_x_2strike", "bins": 8, "shrink": 400., "scale": .5, "weight": .30},
    {"type": "pair", "column": "pitcher_season_success_s50", "context": "batter_hand", "bins": 4, "shrink": 400., "scale": .5, "weight": .20},
    {"type": "pair", "column": "pitcher_season_success_s100", "context": "batter_hand", "bins": 4, "shrink": 1600., "scale": 1., "weight": .15},
    {"type": "pair", "column": "pitcher_reverse_season_s25", "context": "leverage_bucket", "bins": 4, "shrink": 100., "scale": .5, "weight": .35},
    {"type": "pair", "column": "pitcher_reverse_x_advantage", "context": "pitcher_hand", "bins": 8, "shrink": 100., "scale": .5, "weight": .15},
)

FUTURES_CALIBRATION_POLICY = (
    {"type": "probability_pair", "context": "inning_bucket", "bins": 8,
     "shrink": 100., "scale": .1, "weight": .5},
)


def context_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build only row-local state labels used by the frozen tables."""
    out = pd.DataFrame(index=raw.index)
    balls = raw["balls_before"].to_numpy(np.int16)
    strikes = raw["strikes_before"].to_numpy(np.int16)
    out["count_state"] = balls * 3 + strikes
    out["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    )
    for column in (
        "balls_before", "strikes_before", "batter_hand", "pitcher_hand",
        "num_runners_on",
    ):
        out[column] = raw[column].to_numpy()
    out["hand_matchup_code"] = (
        raw["pitcher_hand"].to_numpy() * 3 + raw["batter_hand"].to_numpy()
    )
    out["base_out_code"] = (
        raw["base_state"].astype("string").fillna("<NA>")
        + ":" + raw["outs_before"].astype(str)
    )
    out["inning_bucket"] = np.minimum(
        raw["inning"].fillna(0).to_numpy(np.int16), 10,
    )
    out["score_bucket"] = np.clip(
        raw["score_diff_pitcher_team"].fillna(0).to_numpy(float), -3, 3,
    )
    out["leverage_bucket"] = np.digitize(
        raw["li"].fillna(0).to_numpy(float), (.5, 1., 2., 4.),
    )
    return out


def _numeric_edges(values: pd.Series, bins: int) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric)
    if finite.sum() < 100 or np.nanmin(numeric[finite]) == np.nanmax(numeric[finite]):
        raise ValueError("Numeric residual table has insufficient variation")
    return np.unique(np.quantile(
        numeric[finite], np.linspace(0., 1., bins + 1)[1:-1],
    )).astype(float).tolist()


def _numeric_codes(values: pd.Series, edges: list[float]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric)
    result = np.zeros(len(values), dtype=np.int32)
    result[finite] = np.searchsorted(
        np.asarray(edges, dtype=float), numeric[finite], side="right",
    ) + 1
    return result


def _text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<NA>")


def _category_codes(values: pd.Series, levels: list[str]) -> np.ndarray:
    result = pd.Index(levels).get_indexer(_text(values)).astype(np.int32) + 1
    result[result < 1] = 0
    return result


def _freeze_values(
    source_codes: np.ndarray, residual: np.ndarray, shrink: float, size: int,
) -> list[float]:
    sums = np.bincount(source_codes, weights=residual, minlength=size)
    counts = np.bincount(source_codes, minlength=size)
    table = sums / (counts + float(shrink))
    table -= float(table[source_codes].mean())
    return table.astype(float).tolist()


def _freeze_feature_table(
    features: pd.DataFrame, context: pd.DataFrame, residual: np.ndarray,
    spec: dict,
) -> dict:
    edges = _numeric_edges(features[spec["column"]], int(spec["bins"]))
    numeric = _numeric_codes(features[spec["column"]], edges)
    result = {
        "type": spec["type"], "column": spec["column"], "edges": edges,
        "effective_weight": float(spec["scale"] * spec["weight"]),
        "shrink": float(spec["shrink"]),
    }
    if spec["type"] == "one_d":
        size = len(edges) + 2
        codes = numeric
    else:
        levels = _text(context[spec["context"]]).unique().astype(str).tolist()
        category = _category_codes(context[spec["context"]], levels)
        width = len(levels) + 1
        codes = numeric * width + category
        size = (len(edges) + 2) * width
        result.update({
            "context": spec["context"], "levels": levels, "width": width,
        })
    result["values"] = _freeze_values(
        codes, np.asarray(residual, dtype=float), float(spec["shrink"]), size,
    )
    return result


def _freeze_probability_table(
    base: np.ndarray, context: pd.DataFrame, residual: np.ndarray, spec: dict,
) -> dict:
    base_series = pd.Series(np.asarray(base, dtype=float), index=context.index)
    edges = _numeric_edges(base_series, int(spec["bins"]))
    probability = _numeric_codes(base_series, edges)
    levels = _text(context[spec["context"]]).unique().astype(str).tolist()
    category = _category_codes(context[spec["context"]], levels)
    width = len(levels) + 1
    codes = probability * width + category
    size = (len(edges) + 2) * width
    return {
        "type": "probability_pair", "context": spec["context"],
        "edges": edges, "levels": levels, "width": width,
        "effective_weight": float(spec["scale"] * spec["weight"]),
        "shrink": float(spec["shrink"]),
        "values": _freeze_values(
            codes, np.asarray(residual, dtype=float), float(spec["shrink"]), size,
        ),
    }


def freeze_regime(
    rows: pd.DataFrame, features: pd.DataFrame, base: np.ndarray,
    target: np.ndarray, policy, calibration_policy=(),
) -> dict:
    rows = rows.reset_index(drop=True)
    features = features.reset_index(drop=True)
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)
    if not (len(rows) == len(features) == len(base) == len(target)):
        raise ValueError("Residual table source arrays do not align")
    context = context_frame(rows)
    residual = target - base
    return {
        "feature_tables": [
            _freeze_feature_table(features, context, residual, spec)
            for spec in policy
        ],
        "probability_tables": [
            _freeze_probability_table(base, context, residual, spec)
            for spec in calibration_policy
        ],
        "source_rows": len(rows),
    }


def _apply_feature_table(
    features: pd.DataFrame, context: pd.DataFrame, table: dict,
) -> np.ndarray:
    numeric = _numeric_codes(features[table["column"]], table["edges"])
    if table["type"] == "one_d":
        codes = numeric
    else:
        category = _category_codes(context[table["context"]], table["levels"])
        codes = numeric * int(table["width"]) + category
    values = np.asarray(table["values"], dtype=float)
    return float(table["effective_weight"]) * values[codes]


def _apply_probability_table(
    base: np.ndarray, context: pd.DataFrame, table: dict,
) -> np.ndarray:
    probability = _numeric_codes(pd.Series(base, index=context.index), table["edges"])
    category = _category_codes(context[table["context"]], table["levels"])
    codes = probability * int(table["width"]) + category
    return float(table["effective_weight"]) * np.asarray(
        table["values"], dtype=float,
    )[codes]


def apply_regime(
    rows: pd.DataFrame, features: pd.DataFrame, base: np.ndarray,
    configuration: dict,
) -> np.ndarray:
    rows = rows.reset_index(drop=True)
    features = features.reset_index(drop=True)
    base = np.asarray(base, dtype=float)
    context = context_frame(rows)
    correction = np.zeros(len(rows), dtype=float)
    for table in configuration["feature_tables"]:
        correction += _apply_feature_table(features, context, table)
    for table in configuration.get("probability_tables", []):
        correction += _apply_probability_table(base, context, table)
    return correction


def apply_temporal_portfolio(
    rows: pd.DataFrame, features: pd.DataFrame, base: np.ndarray,
    configuration: dict,
) -> np.ndarray:
    """Apply train-frozen corrections without using other evaluation rows."""
    base = np.asarray(base, dtype=float)
    correction = np.zeros(len(rows), dtype=float)
    game_type = rows["game_type"].astype(str)
    for label, key in (("R", "regular"), ("F", "futures")):
        active = game_type.eq(label).to_numpy()
        if active.any():
            correction[active] = apply_regime(
                rows.loc[active], features.loc[active], base[active],
                configuration[key],
            )
    return correction
