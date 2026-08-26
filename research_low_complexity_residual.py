"""Train strongly regularized LightGBM residual models on stable row-local signals."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_residual_portfolio_v19 import prepare


RAW = [
    "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
    "outs_before", "run_total_before", "score_diff_pitcher_team",
    "num_runners_on", "home_win_expectancy", "li", "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "count_state", "base_out_state",
]
VARIANTS = {
    "leaves4": dict(num_leaves=4, max_depth=3, min_child_samples=10000,
                    n_estimators=120, learning_rate=.025),
    "leaves8": dict(num_leaves=8, max_depth=4, min_child_samples=10000,
                    n_estimators=120, learning_rate=.025),
    "leaves8_slow": dict(num_leaves=8, max_depth=4, min_child_samples=20000,
                         n_estimators=220, learning_rate=.015),
    "leaves16": dict(num_leaves=16, max_depth=5, min_child_samples=20000,
                     n_estimators=120, learning_rate=.02),
}


def features(frame):
    out = frame[RAW].apply(pd.to_numeric, errors="coerce").copy()
    p1 = out["asof_pitcher_prev1_game_success_rate"]
    p3 = out["asof_pitcher_prev3_game_success_rate"]
    p5 = out["asof_pitcher_prev5_game_success_rate"]
    m1 = out["asof_pitcher_prev1_game_middle_rate"]
    m3 = out["asof_pitcher_prev3_game_middle_rate"]
    m5 = out["asof_pitcher_prev5_game_middle_rate"]
    out["recent_success_mean"] = .2 * p1 + .3 * p3 + .5 * p5
    out["recent_success_short"] = .5 * p1 + .5 * p3
    out["success_trend_1_3"] = p1 - p3
    out["success_trend_3_5"] = p3 - p5
    out["recent_middle_mean"] = .2 * m1 + .3 * m3 + .5 * m5
    out["middle_trend_1_3"] = m1 - m3
    out["middle_trend_3_5"] = m3 - m5
    out["recent_minus_career"] = p5 - out["asof_pitcher_success_rate"]
    out["pitcher_minus_batter"] = (
        out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]
    )
    out["failure_proxy"] = (
        out["asof_pitcher_reverse_rate"] + out["asof_pitcher_middle_rate"]
        + out["asof_pitcher_ball_rate"]
    )
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))
    return out.astype(np.float32)


def fit_predict(x_source, residual, source_regular, x_target, target_regular, params, seed):
    model = lgb.LGBMRegressor(
        objective="regression_l2", verbosity=-1, n_jobs=8,
        reg_alpha=5., reg_lambda=300., max_bin=63,
        colsample_bytree=.8, subsample=.8, subsample_freq=1,
        random_state=seed, force_col_wise=True, **params,
    )
    model.fit(x_source.loc[source_regular], residual[source_regular])
    center = float(model.predict(x_source.loc[source_regular]).mean())
    prediction = np.zeros(len(x_target), dtype=np.float64)
    prediction[target_regular] = model.predict(x_target.loc[target_regular]) - center
    return prediction


def coefficients(y, base, correction):
    uncertainty = float(y.mean() * (1. - y.mean()))
    residual = y - base
    return (float(100000. * np.mean(2. * residual * correction) / uncertainty),
            float(100000. * np.mean(correction * correction) / uncertainty))


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    rows = {year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
            for year in (2023, 2024)}
    x = {year: features(rows[year]) for year in rows}
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        folds[year] = {"y": oof["target"][mask].astype(float),
                       "base": oof["blended"][mask].astype(float)}
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]
    m23, m24 = len(rows[2023]) // 2, len(rows[2024]) // 2
    reports = []
    for name, params in VARIANTS.items():
        correction_23h2 = fit_predict(
            x[2023].iloc[:m23], folds[2023]["residual"][:m23],
            rows[2023].iloc[:m23]["game_type"].eq("R").to_numpy(),
            x[2023].iloc[m23:], rows[2023].iloc[m23:]["game_type"].eq("R").to_numpy(),
            params, 3001,
        )
        correction_24 = fit_predict(
            x[2023], folds[2023]["residual"], rows[2023]["game_type"].eq("R").to_numpy(),
            x[2024], rows[2024]["game_type"].eq("R").to_numpy(), params, 3002,
        )
        correction_24h2 = fit_predict(
            x[2024].iloc[:m24], folds[2024]["residual"][:m24],
            rows[2024].iloc[:m24]["game_type"].eq("R").to_numpy(),
            x[2024].iloc[m24:], rows[2024].iloc[m24:]["game_type"].eq("R").to_numpy(),
            params, 3003,
        )
        pairs = {
            "2023_h2": coefficients(folds[2023]["y"][m23:], folds[2023]["base"][m23:], correction_23h2),
            "2024_h1": coefficients(folds[2024]["y"][:m24], folds[2024]["base"][:m24], correction_24[:m24]),
            "2024_h2": coefficients(folds[2024]["y"][m24:], folds[2024]["base"][m24:], correction_24[m24:]),
            "2024_h1_to_h2": coefficients(folds[2024]["y"][m24:], folds[2024]["base"][m24:], correction_24h2),
        }
        for weight in np.arange(-.5, 1.001, .025):
            gains = {label: weight * linear - weight * weight * quadratic
                     for label, (linear, quadratic) in pairs.items()}
            reports.append({"name": name, "params": params, "weight": float(weight),
                            "gains": gains, "min_transfer": min(gains.values()),
                            "mean_transfer": float(np.mean(list(gains.values())))})
        print(f"Completed {name}", flush=True)
    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True)
    output = root / "research/low_complexity_residual_v19.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:60], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
