"""Chronological sparse hierarchical ridge residuals beyond v26."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy import sparse
from sklearn.linear_model import Ridge

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v24_exhaustive_transfer import encode_categorical, encode_numeric
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime
from v26_pareto_policy import (
    FUTURES_CALIBRATION_POLICY,
    FUTURES_POLICY,
    REGULAR_POLICY,
)


ROOT = Path(__file__).resolve().parent
ALPHAS = (100., 1000., 10000., 100000.)
SCALES = (.05, .10, .25, .50, 1.)
BIN_COUNT = 8
CLIP = (.005, .995)
warnings.filterwarnings("ignore", category=PerformanceWarning)

NUMERIC_COLUMNS = (
    "_base_prediction",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "pitcher_season_success_rate",
    "pitcher_season_success_s25",
    "pitcher_season_success_s100",
    "pitcher_prior_success_rate",
    "pitcher_season_minus_prior",
    "pitcher_season_log_n",
    "asof_batter_success_rate",
    "batter_season_success_rate",
    "batter_season_success_s25",
    "batter_season_success_s100",
    "batter_prior_success_rate",
    "batter_season_minus_prior",
    "batter_season_log_n",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "pitcher_middle_season_s25",
    "pitcher_middle_season_s100",
    "pitcher_middle_season_delta",
    "pitcher_reverse_season_s25",
    "pitcher_reverse_season_s100",
    "pitcher_reverse_season_delta",
    "pitcher_ball_season_s25",
    "pitcher_ball_season_s100",
    "pitcher_strike_season_s25",
    "pitcher_strike_season_s100",
    "li", "home_win_expectancy", "away_win_expectancy",
    "score_diff_pitcher_team", "num_runners_on",
)
STATE_COLUMNS = (
    "balls_before", "strikes_before", "outs_before", "base_state",
    "inning", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "batter_team_id",
)
VARIANTS = {
    "state": STATE_COLUMNS,
    "pitcher": (*STATE_COLUMNS, "pitcher_id"),
    "entities": (*STATE_COLUMNS, "pitcher_id", "batter_id"),
}


def _one_hot(codes, offset, rows, columns):
    row_index = np.arange(rows, dtype=np.int32)
    return row_index, codes.astype(np.int32) + offset, columns


def sparse_design(features, raw, base, source, valid, categorical_columns):
    """Fit source-only encodings and return source/query sparse matrices."""
    base_series = pd.Series(np.asarray(base, dtype=np.float32), index=features.index)
    source_rows, source_cols, query_rows, query_cols = [], [], [], []
    offset = 0
    for column in NUMERIC_COLUMNS:
        values = base_series if column == "_base_prediction" else features[column]
        encoded = encode_numeric(
            values.iloc[source], values.iloc[valid], BIN_COUNT,
        )
        if encoded is None:
            continue
        width = int(max(encoded[0].max(initial=0), encoded[1].max(initial=0)) + 1)
        source_rows.append(np.arange(len(source), dtype=np.int32))
        source_cols.append(encoded[0] + offset)
        query_rows.append(np.arange(len(valid), dtype=np.int32))
        query_cols.append(encoded[1] + offset)
        offset += width
    for column in categorical_columns:
        encoded = encode_categorical(raw[column].iloc[source], raw[column].iloc[valid])
        if encoded is None:
            continue
        width = int(max(encoded[0].max(initial=0), encoded[1].max(initial=0)) + 1)
        source_rows.append(np.arange(len(source), dtype=np.int32))
        source_cols.append(encoded[0] + offset)
        query_rows.append(np.arange(len(valid), dtype=np.int32))
        query_cols.append(encoded[1] + offset)
        offset += width

    source_row = np.concatenate(source_rows)
    source_col = np.concatenate(source_cols)
    query_row = np.concatenate(query_rows)
    query_col = np.concatenate(query_cols)
    x_source = sparse.csr_matrix(
        (np.ones(len(source_row), dtype=np.float32), (source_row, source_col)),
        shape=(len(source), offset), dtype=np.float32,
    )
    x_valid = sparse.csr_matrix(
        (np.ones(len(query_row), dtype=np.float32), (query_row, query_col)),
        shape=(len(valid), offset), dtype=np.float32,
    )
    return x_source, x_valid


def make_transfers(active, year):
    indices = {
        value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in indices.items() for half in (1, 2)
    }
    return (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        ("2024", indices[2023], indices[2024]),
    )


def evaluate_regime(rows, features, y, base, year, regime, policy, calibration):
    active = rows["game_type"].eq(regime).to_numpy()
    transfers = make_transfers(active, year)
    blocks = []
    for label, source, valid in transfers:
        frozen = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], y[source],
            policy, calibration,
        )
        source_v26 = np.clip(
            base[source] + apply_regime(
                rows.iloc[source], features.iloc[source], base[source], frozen,
            ), *CLIP,
        )
        valid_v26 = np.clip(
            base[valid] + apply_regime(
                rows.iloc[valid], features.iloc[valid], base[valid], frozen,
            ), *CLIP,
        )
        blocks.append({
            "label": label, "source": source, "valid": valid,
            "source_residual": y[source] - source_v26,
            "v26": valid_v26,
            "masks": segment_masks(rows.iloc[valid].reset_index(drop=True), label),
        })

    reports = []
    for variant, categorical_columns in VARIANTS.items():
        designs = []
        for block in blocks:
            designs.append(sparse_design(
                features, rows, base, block["source"], block["valid"],
                categorical_columns,
            ))
        for alpha in ALPHAS:
            directions = []
            for block, (x_source, x_valid) in zip(blocks, designs):
                model = Ridge(
                    alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-5,
                )
                model.fit(x_source, block["source_residual"])
                source_mean = float(model.predict(x_source).mean())
                directions.append(model.predict(x_valid) - source_mean)
            for scale in SCALES:
                aggregate = {}
                detail_increment = {}
                detail_absolute = {}
                for block, direction in zip(blocks, directions):
                    valid = block["valid"]
                    candidate = np.clip(block["v26"] + scale * direction, *CLIP)
                    for name, mask in block["masks"].items():
                        increment = bss(y[valid][mask], candidate[mask]) - bss(
                            y[valid][mask], block["v26"][mask],
                        )
                        absolute = bss(y[valid][mask], candidate[mask]) - bss(
                            y[valid][mask], base[valid][mask],
                        )
                        detail_increment[name] = increment
                        detail_absolute[name] = absolute
                    aggregate[block["label"]] = detail_increment[
                        f'{block["label"]}/all'
                    ]
                reports.append({
                    "variant": variant, "alpha": alpha, "scale": scale,
                    "gain_2024_increment": aggregate["2024"],
                    "min_transfer_increment": min(aggregate.values()),
                    "min_all_segment_increment": min(detail_increment.values()),
                    "min_all_segment_absolute": min(detail_absolute.values()),
                    "aggregate_increment": aggregate,
                    "detail_increment": detail_increment,
                    "detail_absolute": detail_absolute,
                })
        print(f"{regime}: fitted sparse variant={variant}", flush=True)
    reports.sort(
        key=lambda item: (
            item["gain_2024_increment"], item["min_transfer_increment"],
        ), reverse=True,
    )
    safe = sorted(
        (
            item for item in reports
            if item["min_transfer_increment"] > 0.
            and item["min_all_segment_absolute"] >= 0.
        ),
        key=lambda item: (
            item["gain_2024_increment"], item["min_all_segment_absolute"],
        ), reverse=True,
    )
    return {"reports": reports, "safe": safe}


def compact(items, limit=30):
    fields = (
        "variant", "alpha", "scale", "gain_2024_increment",
        "min_transfer_increment", "min_all_segment_increment",
        "min_all_segment_absolute",
    )
    return [{key: item[key] for key in fields} for item in items[:limit]]


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(raw, target_series)
    features_all = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features_all, raw)
    features_all = add_state_interactions(features_all)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == value) for value in (2023, 2024)
    ])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    rows = raw.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)

    result = {
        "R": evaluate_regime(
            rows, features, y, base, year, "R", REGULAR_POLICY, (),
        ),
        "F": evaluate_regime(
            rows, features, y, base, year, "F", FUTURES_POLICY,
            FUTURES_CALIBRATION_POLICY,
        ),
    }
    path = ROOT / "research/v27_sparse_residual.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        regime: {
            "safe_count": len(payload["safe"]),
            "top": compact(payload["reports"]),
            "safe": compact(payload["safe"]),
        }
        for regime, payload in result.items()
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
