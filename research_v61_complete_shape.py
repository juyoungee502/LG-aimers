"""Research the two missing public-positive shape components for v61.

Both lookup tables are rebuilt only from this project's official-data OOF
residuals.  Public artifacts were used to audit direction and magnitude, never
as a source of table values, model predictions, or packaged files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
CLIP = (0.005, 0.995)

BATTER_K0 = 20000.0
BATTER_K1 = 2000.0
BATTER_T = -4.5
BATTER_STRENGTH = 0.85
BATTER_PUBLIC_ROW_SD = 0.004857905604392169
BATTER_DIRECTION_CORRELATION = 0.9063809167499408

PITCHER_K0 = 50000.0
PITCHER_LOG_S = 0.4
PITCHER_LOG_STRENGTH = 0.8
PITCHER_LOG_PUBLIC_ROW_SD = 0.0008381682456872405
PITCHER_LOG_DIRECTION_CORRELATION = 0.999560094851847

# Public curves, expressed with u=1 at each audited deployed correction.
BATTER_PUBLIC_LINEAR = 27.187136099020805
BATTER_PUBLIC_CURVATURE = 13.3990848
PITCHER_LOG_PUBLIC_LINEAR = 4.3979 * 0.4
PITCHER_LOG_PUBLIC_CURVATURE = 6.6772 * 0.4**2

# Linear-signal retention inferred from the two already transferred axes:
# batter count 0.88365 and pitcher-hand shape 0.80040.
TRANSFER_RATIO = 0.8420271151403191
PUBLIC_V60 = 1124.9


def row_values(rows: pd.DataFrame, column: str, table: pd.Series) -> np.ndarray:
    return rows[column].map(table).fillna(0.0).to_numpy(float)


def normalize_direction(
    table: pd.Series,
    reference: pd.DataFrame,
    column: str,
    target_sd: float,
    strength: float,
) -> tuple[pd.Series, dict]:
    raw_rows = row_values(reference, column, table)
    raw_sd = float(raw_rows.std())
    if not np.isfinite(raw_sd) or raw_sd <= 0.0:
        raise ValueError(f"degenerate {column} direction")
    magnitude_scale = target_sd * strength / raw_sd
    deployed = table * magnitude_scale
    deployed_rows = raw_rows * magnitude_scale
    return deployed, {
        "raw_row_sd": raw_sd,
        "magnitude_scale": float(magnitude_scale),
        "row_mean": float(deployed_rows.mean()),
        "row_std": float(deployed_rows.std()),
        "row_min": float(deployed_rows.min()),
        "row_max": float(deployed_rows.max()),
    }


def freeze_batter_shape(
    source: pd.DataFrame,
    residual: np.ndarray,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    grouped = pd.DataFrame({
        "batter_id": source["batter_id"].to_numpy(),
        "residual": np.asarray(residual, dtype=float),
    }).groupby("batter_id", sort=True)["residual"].agg(["sum", "size"])
    v0 = grouped["sum"] / (grouped["size"] + BATTER_K0)
    vk = grouped["sum"] / (grouped["size"] + BATTER_K1)
    reference_rows = source if reference is None else reference
    base_rows = row_values(reference_rows, "batter_id", v0)
    low_rows = row_values(reference_rows, "batter_id", vk)
    shape_alpha = float(base_rows.std() / low_rows.std())
    raw_direction = BATTER_T * (shape_alpha * vk - v0)
    deployed, stats = normalize_direction(
        raw_direction, reference_rows, "batter_id",
        BATTER_PUBLIC_ROW_SD, BATTER_STRENGTH,
    )
    stats.update({
        "players": int(len(grouped)), "shape_alpha": shape_alpha,
        "k0": BATTER_K0, "k1": BATTER_K1, "t": BATTER_T,
        "strength": BATTER_STRENGTH,
    })
    return deployed, stats


def freeze_pitcher_log(
    source: pd.DataFrame,
    residual: np.ndarray,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    grouped = pd.DataFrame({
        "pitcher_id": source["pitcher_id"].to_numpy(),
        "residual": np.asarray(residual, dtype=float),
    }).groupby("pitcher_id", sort=True)["residual"].agg(["sum", "size"])
    v0 = grouped["sum"] / (grouped["size"] + PITCHER_K0)
    count = grouped["size"].astype(float)
    raw_n = count - count.mean()
    raw_log = np.log1p(count) - np.log1p(count).mean()
    reference_rows = source if reference is None else reference

    row_v0 = row_values(reference_rows, "pitcher_id", v0)
    row_n = row_values(reference_rows, "pitcher_id", raw_n)
    denominator = float(np.dot(row_v0, row_v0))
    if denominator <= 0.0:
        raise ValueError("degenerate pitcher residual level")
    d_n = raw_n - float(np.dot(row_n, row_v0) / denominator) * v0
    basis = np.stack([
        row_values(reference_rows, "pitcher_id", v0),
        row_values(reference_rows, "pitcher_id", d_n),
    ], axis=1)
    dependent = row_values(reference_rows, "pitcher_id", raw_log)
    beta = np.linalg.lstsq(
        basis - basis.mean(axis=0), dependent - dependent.mean(), rcond=None,
    )[0]
    d_log = raw_log - float(beta[0]) * v0 - float(beta[1]) * d_n
    row_d_log = row_values(reference_rows, "pitcher_id", d_log)
    shape_alpha = float(row_v0.std() / row_d_log.std())
    raw_direction = PITCHER_LOG_S * shape_alpha * d_log
    deployed, stats = normalize_direction(
        raw_direction, reference_rows, "pitcher_id",
        PITCHER_LOG_PUBLIC_ROW_SD, PITCHER_LOG_STRENGTH,
    )
    stats.update({
        "players": int(len(grouped)), "shape_alpha": shape_alpha,
        "orthogonal_beta": [float(value) for value in beta],
        "k0": PITCHER_K0, "s": PITCHER_LOG_S,
        "strength": PITCHER_LOG_STRENGTH,
    })
    return deployed, stats


def expected_gain() -> dict:
    batter = (
        BATTER_PUBLIC_LINEAR * TRANSFER_RATIO * BATTER_STRENGTH
        - BATTER_PUBLIC_CURVATURE * BATTER_STRENGTH**2
    )
    pitcher = (
        PITCHER_LOG_PUBLIC_LINEAR * TRANSFER_RATIO * PITCHER_LOG_STRENGTH
        - PITCHER_LOG_PUBLIC_CURVATURE * PITCHER_LOG_STRENGTH**2
    )
    return {
        "batter_shape": float(batter), "pitcher_log": float(pitcher),
        "total": float(batter + pitcher),
        "score": float(PUBLIC_V60 + batter + pitcher),
    }


def correction_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_id", "batter_hand", "control_success",
    ]
    train = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    positions = np.concatenate([
        np.flatnonzero(train["season"].to_numpy(int) == year)
        for year in (2023, 2024)
    ])
    rows = train.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v60_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v60 OOF and training rows are not aligned")
    residual = target - base
    active_2024 = season == 2024
    reference = rows.loc[active_2024].reset_index(drop=True)

    batter, batter_stats = freeze_batter_shape(rows, residual, reference)
    pitcher, pitcher_stats = freeze_pitcher_log(rows, residual, reference)
    batter_rows = row_values(rows, "batter_id", batter)
    pitcher_rows = row_values(rows, "pitcher_id", pitcher)
    production_correction = batter_rows + pitcher_rows
    production_gain = float(
        bss(target[active_2024], np.clip(
            base[active_2024] + production_correction[active_2024], *CLIP,
        )) - bss(target[active_2024], base[active_2024])
    )

    transfers = {}
    for source_year, validation_year in ((2023, 2024), (2024, 2023)):
        source_mask = season == source_year
        validation_mask = season == validation_year
        source_rows = rows.loc[source_mask].reset_index(drop=True)
        validation_rows = rows.loc[validation_mask].reset_index(drop=True)
        batter_table, _ = freeze_batter_shape(source_rows, residual[source_mask])
        pitcher_table, _ = freeze_pitcher_log(source_rows, residual[source_mask])
        batter_delta = row_values(validation_rows, "batter_id", batter_table)
        pitcher_delta = row_values(validation_rows, "pitcher_id", pitcher_table)
        combined = batter_delta + pitcher_delta
        transfers[f"{source_year}_to_{validation_year}"] = {
            "batter_gain": float(bss(
                target[validation_mask], np.clip(base[validation_mask] + batter_delta, *CLIP),
            ) - bss(target[validation_mask], base[validation_mask])),
            "pitcher_log_gain": float(bss(
                target[validation_mask], np.clip(base[validation_mask] + pitcher_delta, *CLIP),
            ) - bss(target[validation_mask], base[validation_mask])),
            "combined_gain": float(bss(
                target[validation_mask], np.clip(base[validation_mask] + combined, *CLIP),
            ) - bss(target[validation_mask], base[validation_mask])),
            "combined_row_std": float(combined.std()),
            "batter_unknown_fraction": float(np.mean(batter_delta == 0.0)),
            "pitcher_unknown_fraction": float(np.mean(pitcher_delta == 0.0)),
        }

    report = {
        "baseline": "v60_public_hand_shape",
        "production": {
            "batter_shape": batter_stats, "pitcher_log": pitcher_stats,
            "combined_row_mean": float(production_correction[active_2024].mean()),
            "combined_row_std": float(production_correction[active_2024].std()),
            "component_correlation": correction_correlation(
                batter_rows[active_2024], pitcher_rows[active_2024],
            ),
            "in_sample_2024_gain": production_gain,
        },
        "transfers": transfers,
        "public_evidence": {
            "batter_shape_reported_gain": 13.788051299020804,
            "batter_direction_correlation": BATTER_DIRECTION_CORRELATION,
            "pitcher_log_reported_gain": 0.6908,
            "pitcher_log_direction_correlation": PITCHER_LOG_DIRECTION_CORRELATION,
            "observed_linear_transfer_ratio": TRANSFER_RATIO,
        },
        "projection": expected_gain(),
        "projected_public_range": [1131.0, 1138.0],
        "external_model_or_prediction_used_in_tables": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v61_complete_shape.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
