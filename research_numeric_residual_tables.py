"""Screen frozen numeric-bin residual tables on four forward-time transfers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_residual_portfolio_v19 import prepare


RAW_FEATURES = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "home_win_expectancy",
    "li",
    "score_diff_pitcher_team",
    "run_total_before",
]
LOG_FEATURES = {
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "li",
}
CONTEXTS = ("none", "count", "hands", "count_hands", "baseout")
BIN_COUNTS = (8, 16)
SHRINKS = (50., 200., 800., 3200.)
WEIGHTS = np.arange(-.5, .801, .025)


def numeric(frame, name, base=None):
    if name == "base_prediction":
        return np.asarray(base, dtype=np.float64)
    value = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    if name in LOG_FEATURES:
        value = np.log1p(np.maximum(value, 0.))
    return value


def fit_edges(value, n_bins):
    valid = value[np.isfinite(value)]
    if not len(valid):
        return np.array([], dtype=np.float64)
    edges = np.unique(np.quantile(valid, np.linspace(0., 1., n_bins + 1)[1:-1]))
    return edges.astype(np.float64)


def bin_value(value, edges):
    result = np.zeros(len(value), dtype=np.int32)
    valid = np.isfinite(value)
    result[valid] = np.searchsorted(edges, value[valid], side="right") + 1
    return result


def context_code(frame, context):
    count = frame["count_state"].to_numpy(np.int32)
    phand = frame["pitcher_hand"].fillna("?").eq("L").to_numpy(np.int32)
    bhand = frame["batter_hand"].fillna("?").eq("L").to_numpy(np.int32)
    if context == "none":
        return np.zeros(len(frame), dtype=np.int32), 1
    if context == "count":
        return count, 12
    if context == "hands":
        return phand * 2 + bhand, 4
    if context == "count_hands":
        return count * 4 + phand * 2 + bhand, 48
    if context == "baseout":
        return frame["base_out_state"].to_numpy(np.int32), 24
    raise ValueError(context)


def fit_effect(frame, residual, feature, base, context, n_bins, shrink):
    value = numeric(frame, feature, base)
    edges = fit_edges(value, n_bins)
    bins = bin_value(value, edges)
    ctx, width = context_code(frame, context)
    code = bins * width + ctx
    size = int((len(edges) + 2) * width)
    sums = np.bincount(code, weights=residual, minlength=size)
    counts = np.bincount(code, minlength=size)
    table = sums / (counts + shrink)
    source_effect = table[code]
    table -= float(source_effect.mean())
    return edges, table, width


def apply_effect(frame, feature, base, context, fitted):
    edges, table, width = fitted
    bins = bin_value(numeric(frame, feature, base), edges)
    ctx, actual_width = context_code(frame, context)
    if width != actual_width:
        raise ValueError("context width differs")
    code = bins * width + ctx
    effect = table[np.minimum(code, len(table) - 1)]
    return np.where(frame["game_type"].eq("R").to_numpy(), effect, 0.)


def coefficients(y, base, value):
    residual = y - base
    uncertainty = float(y.mean() * (1. - y.mean()))
    linear = 100000. * np.mean(2. * residual * value) / uncertainty
    quadratic = 100000. * np.mean(value * value) / uncertainty
    return float(linear), float(quadratic)


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    rows = {year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
            for year in (2023, 2024)}
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
        }
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]

    m23, m24 = len(rows[2023]) // 2, len(rows[2024]) // 2
    sources = {
        "23h1": (rows[2023].iloc[:m23], folds[2023]["residual"][:m23],
                  folds[2023]["base"][:m23]),
        "23": (rows[2023], folds[2023]["residual"], folds[2023]["base"]),
        "24h1": (rows[2024].iloc[:m24], folds[2024]["residual"][:m24],
                  folds[2024]["base"][:m24]),
    }
    blocks = {
        "2023_h2": (rows[2023].iloc[m23:], folds[2023]["y"][m23:],
                    folds[2023]["base"][m23:], "23h1"),
        "2024_h1": (rows[2024].iloc[:m24], folds[2024]["y"][:m24],
                    folds[2024]["base"][:m24], "23"),
        "2024_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                    folds[2024]["base"][m24:], "23"),
        "2024_h1_to_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                           folds[2024]["base"][m24:], "24h1"),
    }
    reports = []
    for feature in RAW_FEATURES + ["base_prediction"]:
        for context in CONTEXTS:
            for n_bins in BIN_COUNTS:
                for shrink in SHRINKS:
                    fitted = {}
                    for label, (frame, residual, base) in sources.items():
                        fitted[label] = fit_effect(
                            frame, residual, feature, base, context, n_bins, shrink,
                        )
                    pairs = {}
                    for label, (frame, y, base, source) in blocks.items():
                        value = apply_effect(
                            frame, feature, base, context, fitted[source],
                        )
                        pairs[label] = coefficients(y, base, value)
                    for weight in WEIGHTS:
                        gains = {label: weight * linear - weight * weight * quadratic
                                 for label, (linear, quadratic) in pairs.items()}
                        reports.append({
                            "feature": feature, "context": context,
                            "n_bins": n_bins, "shrink": shrink,
                            "weight": float(weight), "gains": gains,
                            "min_transfer": min(gains.values()),
                            "mean_transfer": float(np.mean(list(gains.values()))),
                        })
    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]),
                 reverse=True)
    output = root / "research/numeric_residual_tables_v19.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:100], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
