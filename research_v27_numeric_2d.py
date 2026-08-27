"""Screen exposure-aware two-dimensional residual tables beyond v26.

The same observed rate has very different uncertainty at 20 and 2,000 prior
events.  These experiments therefore cross a rate-like feature with a strictly
pre-pitch exposure feature.  Every table is fitted on an earlier time block,
frozen, and applied to a later block.  Evaluation rows are never aggregated.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v24_exhaustive_transfer import encode_numeric, table_direction
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime
from v26_pareto_policy import (
    FUTURES_CALIBRATION_POLICY,
    FUTURES_POLICY,
    REGULAR_POLICY,
)


ROOT = Path(__file__).resolve().parent
BIN_PAIRS = ((4, 4), (8, 4), (8, 8), (16, 4), (16, 8))
SHRINKS = (100., 400., 1600., 6400.)
SCALES = (.10, .25, .50, 1.00)
CLIP = (.005, .995)
warnings.filterwarnings("ignore", category=PerformanceWarning)


# The second feature is always a row-local, pre-pitch measure of exposure.
PAIR_SPECS = (
    ("asof_pitcher_success_rate", "log1p_asof_pitcher_n"),
    ("asof_pitcher_prev1_game_success_rate", "pitcher_season_log_n"),
    ("asof_pitcher_prev3_game_success_rate", "pitcher_season_log_n"),
    ("asof_pitcher_prev5_game_success_rate", "pitcher_season_log_n"),
    ("pitcher_season_success_rate", "pitcher_season_log_n"),
    ("pitcher_season_success_s25", "pitcher_season_log_n"),
    ("pitcher_season_success_s100", "pitcher_season_log_n"),
    ("pitcher_season_minus_prior", "pitcher_season_log_n"),
    ("pitcher_success_trend_1_3", "pitcher_season_log_n"),
    ("pitcher_success_trend_1_5", "pitcher_season_log_n"),
    ("asof_batter_success_rate", "log1p_asof_batter_n"),
    ("batter_season_success_rate", "batter_season_log_n"),
    ("batter_season_success_s25", "batter_season_log_n"),
    ("batter_season_success_s100", "batter_season_log_n"),
    ("batter_season_minus_prior", "batter_season_log_n"),
    ("asof_pitcher_middle_rate", "log1p_asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "log1p_asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "log1p_asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "log1p_asof_pitcher_n"),
    ("asof_pitcher_fastball_rate", "log1p_asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "log1p_asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "log1p_asof_pitcher_pitchmix_n"),
    ("pitcher_middle_season_rate", "pitcher_season_log_n"),
    ("pitcher_middle_season_s25", "pitcher_season_log_n"),
    ("pitcher_middle_season_s100", "pitcher_season_log_n"),
    ("pitcher_middle_season_delta", "pitcher_season_log_n"),
    ("pitcher_reverse_season_rate", "pitcher_season_log_n"),
    ("pitcher_reverse_season_s25", "pitcher_season_log_n"),
    ("pitcher_reverse_season_s100", "pitcher_season_log_n"),
    ("pitcher_reverse_season_delta", "pitcher_season_log_n"),
    ("pitcher_ball_season_rate", "pitcher_season_log_n"),
    ("pitcher_ball_season_s25", "pitcher_season_log_n"),
    ("pitcher_ball_season_s100", "pitcher_season_log_n"),
    ("pitcher_strike_season_rate", "pitcher_season_log_n"),
    ("pitcher_strike_season_s25", "pitcher_season_log_n"),
    ("pitcher_strike_season_s100", "pitcher_season_log_n"),
    ("asof_batter_middle_rate", "log1p_asof_batter_n"),
    ("batter_middle_season_rate", "batter_season_log_n"),
    ("batter_middle_season_s25", "batter_season_log_n"),
    ("batter_middle_season_s100", "batter_season_log_n"),
)


def numeric_2d_codes(source_x, query_x, source_y, query_y, bins_x, bins_y):
    x_codes = encode_numeric(source_x, query_x, bins_x)
    y_codes = encode_numeric(source_y, query_y, bins_y)
    if x_codes is None or y_codes is None:
        return None
    width = int(max(y_codes[0].max(initial=0), y_codes[1].max(initial=0)) + 1)
    return (
        x_codes[0] * width + y_codes[0],
        x_codes[1] * width + y_codes[1],
    )


def make_transfers(active, year):
    indices = {
        value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index) // 2] if half == 1 else index[len(index) // 2:]
        for value, index in indices.items() for half in (1, 2)
    }
    return indices, (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        ("2024", indices[2023], indices[2024]),
    )


def prepare_blocks(rows, features, y, base, year, regime, policy, calibration):
    active = rows["game_type"].eq(regime).to_numpy()
    _indices, transfers = make_transfers(active, year)
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
    return blocks


def evaluate_regime(rows, features, y, base, year, regime, policy, calibration):
    blocks = prepare_blocks(
        rows, features, y, base, year, regime, policy, calibration,
    )
    baseline = {}
    for block in blocks:
        valid = block["valid"]
        for name, mask in block["masks"].items():
            baseline[name] = bss(y[valid][mask], block["v26"][mask]) - bss(
                y[valid][mask], base[valid][mask],
            )

    approximate = []
    for spec_index, (x_column, y_column) in enumerate(PAIR_SPECS):
        for bins_x, bins_y in BIN_PAIRS:
            encoded = []
            for block in blocks:
                source, valid = block["source"], block["valid"]
                codes = numeric_2d_codes(
                    features[x_column].iloc[source], features[x_column].iloc[valid],
                    features[y_column].iloc[source], features[y_column].iloc[valid],
                    bins_x, bins_y,
                )
                if codes is None:
                    break
                encoded.append(codes)
            if len(encoded) != len(blocks):
                continue
            for shrink in SHRINKS:
                directions = [
                    table_direction(codes[0], codes[1], block["source_residual"], shrink)
                    for block, codes in zip(blocks, encoded)
                ]
                for scale in SCALES:
                    aggregate_increment = {}
                    for block, direction in zip(blocks, directions):
                        valid = block["valid"]
                        residual = y[valid] - block["v26"]
                        reference = float(y[valid].mean() * (1. - y[valid].mean()))
                        aggregate_increment[block["label"]] = float(
                            100000. * (
                                2. * scale * np.mean(residual * direction)
                                - scale * scale * np.mean(direction * direction)
                            ) / reference
                        )
                    if min(aggregate_increment.values()) <= 0:
                        continue
                    approximate.append({
                        "x": x_column, "exposure": y_column,
                        "bins_x": bins_x, "bins_exposure": bins_y,
                        "shrink": shrink, "scale": scale,
                        "aggregate_increment": aggregate_increment,
                        "gain_2024_increment": aggregate_increment["2024"],
                        "min_transfer_increment": min(aggregate_increment.values()),
                    })
        print(
            f"{regime}: screened {spec_index + 1}/{len(PAIR_SPECS)} pairs",
            flush=True,
        )
    approximate.sort(
        key=lambda item: (
            item["gain_2024_increment"], item["min_transfer_increment"],
        ), reverse=True,
    )

    # Exact clipping and detailed segment scores are relatively expensive.  Use
    # them only on the Pareto-relevant head from the vectorized first stage.
    shortlist_ids = {id(item) for item in approximate[:300]}
    by_floor = sorted(
        approximate, key=lambda item: item["min_transfer_increment"], reverse=True,
    )
    shortlist_ids.update(id(item) for item in by_floor[:300])
    by_balance = sorted(
        approximate,
        key=lambda item: item["gain_2024_increment"]
        + 2. * item["min_transfer_increment"],
        reverse=True,
    )
    shortlist_ids.update(id(item) for item in by_balance[:300])
    shortlisted = [item for item in approximate if id(item) in shortlist_ids]
    grouped = {}
    for item in shortlisted:
        key = (
            item["x"], item["exposure"], item["bins_x"],
            item["bins_exposure"], item["shrink"],
        )
        grouped.setdefault(key, []).append(item)

    reports = []
    for (x_column, y_column, bins_x, bins_y, shrink), items in grouped.items():
        directions = []
        for block in blocks:
            source, valid = block["source"], block["valid"]
            codes = numeric_2d_codes(
                features[x_column].iloc[source], features[x_column].iloc[valid],
                features[y_column].iloc[source], features[y_column].iloc[valid],
                bins_x, bins_y,
            )
            directions.append(table_direction(
                codes[0], codes[1], block["source_residual"], shrink,
            ))
        for item in items:
            scale = float(item["scale"])
            aggregate_increment = {}
            detail_increment = {}
            detail_absolute = {}
            for block, direction in zip(blocks, directions):
                valid = block["valid"]
                candidate = np.clip(block["v26"] + scale * direction, *CLIP)
                all_mask = block["masks"][f'{block["label"]}/all']
                aggregate_increment[block["label"]] = bss(
                    y[valid][all_mask], candidate[all_mask],
                ) - bss(y[valid][all_mask], block["v26"][all_mask])
                for name, mask in block["masks"].items():
                    detail_increment[name] = bss(
                        y[valid][mask], candidate[mask],
                    ) - bss(y[valid][mask], block["v26"][mask])
                    detail_absolute[name] = bss(
                        y[valid][mask], candidate[mask],
                    ) - bss(y[valid][mask], base[valid][mask])
            if min(aggregate_increment.values()) <= 0:
                continue
            reports.append({
                **item,
                "aggregate_increment": aggregate_increment,
                "gain_2024_increment": aggregate_increment["2024"],
                "min_transfer_increment": min(aggregate_increment.values()),
                "min_all_segment_increment": min(detail_increment.values()),
                "min_all_segment_absolute": min(detail_absolute.values()),
                "min_2024_segment_increment": min(
                    value for name, value in detail_increment.items()
                    if name.startswith("2024/")
                ),
                "min_2024_segment_absolute": min(
                    value for name, value in detail_absolute.items()
                    if name.startswith("2024/")
                ),
                "detail_2024_increment": detail_increment,
                "detail_2024_absolute": detail_absolute,
            })
    reports.sort(
        key=lambda item: (
            item["gain_2024_increment"], item["min_transfer_increment"],
            item["min_2024_segment_increment"],
        ), reverse=True,
    )
    robust = sorted(
        (
            item for item in reports
            if item["min_all_segment_absolute"] >= 0.
        ),
        key=lambda item: (
            item["gain_2024_increment"], item["min_all_segment_absolute"],
        ), reverse=True,
    )
    return {
        "v26_baseline": baseline,
        "positive_approximate_count": len(approximate),
        "exact_shortlist_count": len(shortlisted),
        "positive_transfer": reports,
        "robust": robust,
    }


def compact(items, limit=30):
    fields = (
        "x", "exposure", "bins_x", "bins_exposure", "shrink", "scale",
        "gain_2024_increment", "min_transfer_increment",
        "min_all_segment_increment", "min_all_segment_absolute",
        "min_2024_segment_increment", "min_2024_segment_absolute",
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
    path = ROOT / "research/v27_numeric_2d.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        regime: {
            "v26_min_absolute": min(report["v26_baseline"].values()),
            "positive_approximate_count": report["positive_approximate_count"],
            "exact_shortlist_count": report["exact_shortlist_count"],
            "positive_transfer_count": len(report["positive_transfer"]),
            "robust_count": len(report["robust"]),
            "top_positive": compact(report["positive_transfer"]),
            "top_robust": compact(report["robust"]),
        }
        for regime, report in result.items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
