"""Low-dimensional auxiliary meta correction transferred from 2023 to 2024.

The screen connects the v29 failure-shape heads and the diverse v30 residual
channels to the strong v23 base.  Coefficients are learned only on 2023 OOF
residuals and transferred unchanged to 2024; no 2024 target enters fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
ALPHAS = (100., 300., 1000., 3000., 10000., 30000., 100000., 300000.)
SCALES = np.round(np.arange(0., .501, .025), 3)
RAW_NUMERIC = (
    "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def groups(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "R": rows["game_type"].eq("R").to_numpy(),
        "F": rows["game_type"].eq("F").to_numpy(),
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
    }


def audit(target, base, correction, rows):
    masks = groups(rows)
    base_scores = {
        name: bss(target[active], base[active])
        for name, active in masks.items()
    }
    result = []
    for scale in SCALES:
        prediction = np.clip(base + scale * correction, .005, .995)
        gains = {
            name: bss(target[active], prediction[active]) - base_scores[name]
            for name, active in masks.items()
        }
        result.append({
            "scale": float(scale), "gains": gains,
            "score": base_scores["all"] + gains["all"],
            "worst_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
            "worst_type": min(gains["R"], gains["F"]),
        })
    return result


def year_data(raw, archive, year):
    raw_active = raw["season"].eq(year).to_numpy()
    oof_active = archive["season"] == year
    rows = raw.loc[raw_active].reset_index(drop=True)
    target = archive["target"][oof_active].astype(float)
    base = archive["blended"][oof_active].astype(float)
    with np.load(ROOT / f"research/v29_rmo_multitask_{year}.npz") as source:
        rmo = source["meta"].astype(float)
        hazard = source["hazard_success"].astype(float)[:, None]
        joint = source["joint_success"].astype(float)[:, None]
    with np.load(
        ROOT / f"research/v30_hierarchical_residual_{year}.npz"
    ) as source:
        v30_base = source["base"].astype(float)
        individual = v30_base[:, None] + source["corrections"].astype(float)
        combined = source["predictions"].astype(float)
    if not np.allclose(target, raw.loc[raw_active, "control_success"]):
        raise ValueError(f"row alignment failed for {year}")

    disagreement = np.column_stack([
        individual - base[:, None], combined - base[:, None],
        individual[:, :3].std(axis=1),
        individual[:, :3].max(axis=1) - individual[:, :3].min(axis=1),
        v30_base - base,
    ])
    auxiliary = np.column_stack([rmo, hazard, joint])
    numeric = rows[list(RAW_NUMERIC)].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(float)
    numeric[:, 0] = np.log1p(np.maximum(numeric[:, 0], 0.))
    exposure = np.column_stack([
        np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce")),
        np.log1p(pd.to_numeric(rows["asof_batter_n"], errors="coerce")),
        np.log1p(pd.to_numeric(
            rows["asof_pitcher_pitchmix_n"], errors="coerce",
        )),
    ])
    categorical = pd.DataFrame({
        "game_type": rows["game_type"].astype(str),
        "count": (
            rows["balls_before"].astype(str) + "-"
            + rows["strikes_before"].astype(str)
        ),
        "hand": (
            rows["pitcher_hand"].astype(str) + "-"
            + rows["batter_hand"].astype(str)
        ),
        "base": rows["base_state"].astype(str),
        "top": rows["top_bottom"].astype(str),
    })
    return {
        "rows": rows, "target": target, "base": base,
        "disagreement": disagreement, "auxiliary": auxiliary,
        "numeric": np.column_stack([numeric, exposure]),
        "categorical": categorical,
    }


def standardized(train, valid):
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[~np.isfinite(std) | (std < 1e-7)] = 1.
    return (
        np.nan_to_num((train - mean) / std),
        np.nan_to_num((valid - mean) / std),
    )


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as source:
        archive = {key: source[key] for key in source.files}
    data = {year: year_data(raw, archive, year) for year in (2023, 2024)}
    dummies = pd.get_dummies(pd.concat([
        data[2023]["categorical"], data[2024]["categorical"],
    ], ignore_index=True), dtype=np.float32).to_numpy(float)
    split = len(data[2023]["rows"])
    data[2023]["categorical_matrix"] = dummies[:split]
    data[2024]["categorical_matrix"] = dummies[split:]

    variants = {
        "disagreement": ("disagreement",),
        "auxiliary": ("auxiliary",),
        "disagreement_aux": ("disagreement", "auxiliary"),
        "full_numeric": (
            "disagreement", "auxiliary", "numeric", "categorical_matrix",
        ),
    }
    reports = {}
    stored = {}
    train_regular = data[2023]["rows"]["game_type"].eq("R").to_numpy()
    valid_regular = data[2024]["rows"]["game_type"].eq("R").to_numpy()
    for variant, pieces in variants.items():
        x_train = np.column_stack([data[2023][name] for name in pieces])
        x_valid = np.column_stack([data[2024][name] for name in pieces])
        x_train, x_valid = standardized(x_train, x_valid)
        residual = data[2023]["target"] - data[2023]["base"]
        reports[variant] = []
        for segmented in (False, True):
            for fit_intercept in (False, True):
                for alpha in ALPHAS:
                    correction = np.zeros(len(x_valid), dtype=float)
                    segment_values = (True, False) if segmented else (None,)
                    for segment_value in segment_values:
                        if segment_value is None:
                            train = np.ones(len(x_train), dtype=bool)
                            valid = np.ones(len(x_valid), dtype=bool)
                        else:
                            train = train_regular == segment_value
                            valid = valid_regular == segment_value
                        model = Ridge(
                            alpha=alpha, fit_intercept=fit_intercept,
                        ).fit(x_train[train], residual[train])
                        correction[valid] = model.predict(x_valid[valid])
                    audits = audit(
                        data[2024]["target"], data[2024]["base"],
                        correction, data[2024]["rows"],
                    )
                    best = max(audits, key=lambda row: row["gains"]["all"])
                    safe = max(audits, key=lambda row: (
                        min(row["worst_quarter"], row["worst_type"]),
                        row["gains"]["all"],
                    ))
                    reports[variant].append({
                        "segmented": segmented,
                        "fit_intercept": fit_intercept,
                        "alpha": alpha,
                        "best": best,
                        "safe": safe,
                    })
                    stored[
                        f"{variant}_{int(segmented)}_{int(fit_intercept)}_"
                        f"{int(alpha)}"
                    ] = correction.astype(np.float32)
        reports[variant].sort(
            key=lambda row: (
                min(row["safe"]["worst_quarter"],
                    row["safe"]["worst_type"]),
                row["best"]["gains"]["all"],
            ), reverse=True,
        )
        top = reports[variant][0]
        print(variant, json.dumps(top, indent=2), flush=True)

    (ROOT / "research/v31_linear_meta_transfer.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "research/v31_linear_meta_transfer.npz", **stored,
    )
    print("Saved v31 linear meta-transfer audit", flush=True)


if __name__ == "__main__":
    main()
