"""Screen two-dimensional frozen residual tables across forward transfers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_numeric_residual_tables import context_code, numeric
from research_residual_portfolio_v19 import prepare


PAIRS = {
    "success_1_3": ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate"),
    "success_1_5": ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate"),
    "success_3_5": ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate"),
    "success_middle_1": ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"),
    "success_middle_3": ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev3_game_middle_rate"),
    "success_middle_5": ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate"),
    "recent_batter": ("derived_recent_success_mean", "asof_batter_success_rate"),
    "career_recent": ("asof_pitcher_success_rate", "derived_recent_success_mean"),
    "reverse_middle": ("asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"),
    "fastball_breaking": ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate"),
}
CONTEXTS = ("none", "count", "baseout")


def edges(value, n_bins):
    valid = value[np.isfinite(value)]
    return np.unique(np.quantile(valid, np.linspace(0., 1., n_bins + 1)[1:-1])) \
        if len(valid) else np.array([], dtype=np.float64)


def bins(value, fitted):
    output = np.zeros(len(value), dtype=np.int32)
    valid = np.isfinite(value)
    output[valid] = np.searchsorted(fitted, value[valid], side="right") + 1
    return output


def fit_pair(frame, residual, base, pair, context, n_bins, shrink):
    first, second = (numeric(frame, feature, base) for feature in pair)
    e1, e2 = edges(first, n_bins), edges(second, n_bins)
    b1, b2 = bins(first, e1), bins(second, e2)
    card2 = len(e2) + 2
    ctx, width = context_code(frame, context)
    code = (b1 * card2 + b2) * width + ctx
    size = (len(e1) + 2) * card2 * width
    sums = np.bincount(code, weights=residual, minlength=size)
    counts = np.bincount(code, minlength=size)
    table = sums / (counts + shrink)
    table -= float(table[code].mean())
    return e1, e2, table, width


def apply_pair(frame, base, pair, context, fitted):
    e1, e2, table, width = fitted
    first, second = (numeric(frame, feature, base) for feature in pair)
    ctx, actual_width = context_code(frame, context)
    if width != actual_width:
        raise ValueError("Pair context width differs")
    code = (bins(first, e1) * (len(e2) + 2) + bins(second, e2)) * width + ctx
    value = table[np.minimum(code, len(table) - 1)]
    return np.where(frame["game_type"].eq("R").to_numpy(), value, 0.)


def coefficients(y, base, value):
    uncertainty = float(y.mean() * (1. - y.mean()))
    residual = y - base
    return (float(100000. * np.mean(2. * residual * value) / uncertainty),
            float(100000. * np.mean(value * value) / uncertainty))


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
        folds[year] = {"y": oof["target"][mask].astype(np.float64),
                       "base": oof["blended"][mask].astype(np.float64)}
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]
    m23, m24 = len(rows[2023]) // 2, len(rows[2024]) // 2
    sources = {
        "23h1": (rows[2023].iloc[:m23], folds[2023]["residual"][:m23], folds[2023]["base"][:m23]),
        "23": (rows[2023], folds[2023]["residual"], folds[2023]["base"]),
        "24h1": (rows[2024].iloc[:m24], folds[2024]["residual"][:m24], folds[2024]["base"][:m24]),
    }
    blocks = {
        "2023_h2": (rows[2023].iloc[m23:], folds[2023]["y"][m23:], folds[2023]["base"][m23:], "23h1"),
        "2024_h1": (rows[2024].iloc[:m24], folds[2024]["y"][:m24], folds[2024]["base"][:m24], "23"),
        "2024_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:], folds[2024]["base"][m24:], "23"),
        "2024_h1_to_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:], folds[2024]["base"][m24:], "24h1"),
    }
    reports = []
    for name, pair in PAIRS.items():
        for context in CONTEXTS:
            for n_bins in (4, 8):
                for shrink in (50., 200., 800., 3200.):
                    fitted = {label: fit_pair(frame, residual, base, pair, context, n_bins, shrink)
                              for label, (frame, residual, base) in sources.items()}
                    pairs = {label: coefficients(y, base, apply_pair(frame, base, pair, context, fitted[source]))
                             for label, (frame, y, base, source) in blocks.items()}
                    for weight in np.arange(-.5, .801, .025):
                        gains = {label: weight * linear - weight * weight * quadratic
                                 for label, (linear, quadratic) in pairs.items()}
                        reports.append({"name": name, "features": list(pair), "context": context,
                                        "n_bins": n_bins, "shrink": shrink, "weight": float(weight),
                                        "gains": gains, "min_transfer": min(gains.values()),
                                        "mean_transfer": float(np.mean(list(gains.values())))})
    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True)
    output = root / "research/pair_residual_tables_v19.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:100], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
