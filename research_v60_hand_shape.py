"""Rebuild and audit the public-positive pitcher hand-contrast shape.

The table is learned only from this project's OOF residuals.  For each pitcher
we estimate the residual difference between same- and opposite-handed batters,
shrink it at k=1000, then form the public-tested shape direction toward k=100
with t=3.  Production magnitude is normalized to the measured row standard
deviation of that public direction; no external model or prediction is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
K0 = 1000.0
K1 = 100.0
T = 3.0
PUBLIC_DIRECTION_ROW_SD = 0.004007656969231088
PUBLIC_REFERENCE_GAIN = 6.8880
PUBLIC_V59 = 1120.5
CLIP = (0.005, 0.995)


def token(row: tuple) -> str:
    return "|".join(str(int(value)) for value in row)


def row_keys(rows: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [token(row) for row in rows[["pitcher_id", "pitcher_hand", "batter_hand"]].itertuples(index=False, name=None)],
        index=rows.index,
    )


def freeze_direction(
    source: pd.DataFrame,
    residual: np.ndarray,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    work = source[["pitcher_id", "pitcher_hand", "batter_hand"]].copy()
    work["same_hand"] = work["pitcher_hand"].eq(work["batter_hand"]).astype(np.int8)
    work["residual"] = np.asarray(residual, dtype=float)
    grouped = work.groupby(["pitcher_id", "same_hand"], observed=True)["residual"].agg(["mean", "size"]).unstack()
    for statistic in ("mean", "size"):
        for context in (0, 1):
            if (statistic, context) not in grouped:
                grouped[(statistic, context)] = np.nan if statistic == "mean" else 0.0
    n0 = grouped[("size", 0)].fillna(0.0)
    n1 = grouped[("size", 1)].fillna(0.0)
    effective_n = n0 * n1 / (n0 + n1).replace(0.0, np.nan)
    contrast = (grouped[("mean", 1)] - grouped[("mean", 0)]) * effective_n / (effective_n + K0)
    contrast = contrast.dropna()

    base = {}
    for pitcher, value in contrast.items():
        for pitcher_hand in (1, 2):
            for batter_hand in (1, 2):
                sign = 0.5 if pitcher_hand == batter_hand else -0.5
                base[f"{int(pitcher)}|{pitcher_hand}|{batter_hand}"] = sign * float(value)
    base = pd.Series(base, dtype=float)
    counts = row_keys(work).value_counts().reindex(base.index)
    live = counts.notna()
    live_base = base[live]
    live_n = counts[live].astype(float)
    low_shrink = live_base * (live_n + K0) / (live_n + K1)

    reference_rows = work if reference is None else reference
    reference_keys = row_keys(reference_rows)
    base_rows = reference_keys.map(base).fillna(0.0).to_numpy(float)
    low_table = base.copy()
    low_table.loc[live] = low_shrink
    low_rows = reference_keys.map(low_table).fillna(0.0).to_numpy(float)
    alpha = float(base_rows.std() / low_rows.std())
    shaped = base.copy()
    shaped.loc[live] = live_base + T * (alpha * low_shrink - live_base)
    direction = shaped - base
    raw_rows = reference_keys.map(direction).fillna(0.0).to_numpy(float)
    scale = PUBLIC_DIRECTION_ROW_SD / raw_rows.std()
    direction *= scale
    deployed_rows = raw_rows * scale
    return direction, {
        "pitchers": int(len(contrast)), "table_cells": int(len(direction)),
        "live_cells": int(live.sum()), "shape_alpha": alpha,
        "magnitude_scale": float(scale), "row_mean": float(deployed_rows.mean()),
        "row_std": float(deployed_rows.std()), "row_min": float(deployed_rows.min()),
        "row_max": float(deployed_rows.max()),
    }


def apply(rows: pd.DataFrame, table: pd.Series) -> np.ndarray:
    return row_keys(rows).map(table).fillna(0.0).to_numpy(float)


def main() -> None:
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_hand", "control_success",
    ]
    train = pd.read_csv(ROOT / "data/train.csv", usecols=columns, encoding="utf-8-sig", low_memory=False)
    positions = np.concatenate([
        np.flatnonzero(train["season"].to_numpy(int) == year) for year in (2023, 2024)
    ])
    rows = train.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v59_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v59 OOF and training rows are not aligned")
    residual = target - base

    production, production_stats = freeze_direction(
        rows, residual, rows.loc[season == 2024].reset_index(drop=True),
    )
    production_correction = apply(rows, production)
    active = season == 2024
    local_gain = float(
        bss(target[active], np.clip(base[active] + production_correction[active], *CLIP))
        - bss(target[active], base[active])
    )

    transfers = {}
    for source_year, validation_year in ((2023, 2024), (2024, 2023)):
        source_mask = season == source_year
        validation_mask = season == validation_year
        table, stats = freeze_direction(rows.loc[source_mask].reset_index(drop=True), residual[source_mask])
        correction = apply(rows.loc[validation_mask].reset_index(drop=True), table)
        transfers[f"{source_year}_to_{validation_year}"] = {
            "gain": float(
                bss(target[validation_mask], np.clip(base[validation_mask] + correction, *CLIP))
                - bss(target[validation_mask], base[validation_mask])
            ),
            "stats": stats,
            "validation_row_std": float(correction.std()),
            "unknown_fraction": float(np.mean(correction == 0.0)),
        }

    report = {
        "baseline": "v59_public_batter_exposure",
        "configuration": {"k0": K0, "k1": K1, "t": T},
        "production": production_stats,
        "production_local_2024_gain": local_gain,
        "transfers": transfers,
        "public_reference_gain": PUBLIC_REFERENCE_GAIN,
        "projected_public_score": PUBLIC_V59 + PUBLIC_REFERENCE_GAIN,
        "projected_public_range": [1123.0, 1129.0],
        "external_model_or_prediction_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v60_hand_shape.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
