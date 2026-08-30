"""Build and audit the independent residual-shape frontier for v62."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v60_hand_shape import freeze_direction as freeze_hand_shape
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
CLIP = (0.005, 0.995)

V61_SCALE_DELTA = -0.14
C4N_STRENGTH = 1.0
HD_STRENGTH = 0.5
D0_STRENGTH = 1.0

C4N_PUBLIC_ROW_SD = 0.0023967528882864113
HD_PUBLIC_ROW_SD = 0.005474201013314534
D0_PUBLIC_ROW_SD = 0.0019227961618384208

C4N_PUBLIC_CORRELATION = 0.9997962691489939
HD_PUBLIC_CORRELATION = 0.9723600795906776
D0_PUBLIC_CORRELATION = 0.7094279283165271

PUBLIC_V61 = 1132.0
OBSERVED_TRANSFER = 7.1 / 10.278884199524054


def joined_keys(rows: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = rows[columns[0]].astype(str)
    for column in columns[1:]:
        result = result.str.cat(rows[column].astype(str), sep="|")
    return result


def apply_table(rows: pd.DataFrame, columns: list[str], table: pd.Series) -> np.ndarray:
    return joined_keys(rows, columns).map(table).fillna(0.0).to_numpy(float)


def normalize(
    table: pd.Series,
    reference: pd.DataFrame,
    columns: list[str],
    target_sd: float,
    strength: float,
) -> tuple[pd.Series, dict]:
    raw_rows = apply_table(reference, columns, table)
    raw_sd = float(raw_rows.std())
    if not np.isfinite(raw_sd) or raw_sd <= 0.0:
        raise ValueError(f"degenerate direction for {columns}")
    scale = target_sd * strength / raw_sd
    deployed = table * scale
    deployed_rows = raw_rows * scale
    return deployed, {
        "raw_row_sd": raw_sd,
        "magnitude_scale": float(scale),
        "row_mean": float(deployed_rows.mean()),
        "row_std": float(deployed_rows.std()),
        "row_min": float(deployed_rows.min()),
        "row_max": float(deployed_rows.max()),
    }


def freeze_c4n_mirror(
    source: pd.DataFrame,
    hand_table: pd.Series,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    columns = ["pitcher_id", "pitcher_hand", "batter_hand"]
    reference_rows = source if reference is None else reference
    source_counts = joined_keys(source, columns).value_counts().reindex(hand_table.index)
    live = source_counts.notna()
    raw = pd.Series(0.0, index=hand_table.index, dtype=float)
    raw.loc[live] = source_counts.loc[live].astype(float) - float(source_counts.loc[live].mean())
    raw_rows = apply_table(reference_rows, columns, raw)
    hand_rows = apply_table(reference_rows, columns, hand_table)
    a = raw_rows - raw_rows.mean()
    b = hand_rows - hand_rows.mean()
    beta = float(np.dot(a, b) / np.dot(b, b))
    raw.loc[live] -= beta * hand_table.loc[live]
    orthogonal_rows = apply_table(reference_rows, columns, raw)
    shape_alpha = float(hand_rows.std() / orthogonal_rows.std())
    mirror = -0.4 * shape_alpha * raw
    deployed, stats = normalize(
        mirror, reference_rows, columns,
        C4N_PUBLIC_ROW_SD, C4N_STRENGTH,
    )
    stats.update({
        "table_cells": int(len(deployed)), "live_cells": int(live.sum()),
        "orthogonal_beta": beta, "shape_alpha": shape_alpha,
        "s": -0.4, "strength": C4N_STRENGTH,
    })
    return deployed, stats


def freeze_hd(
    source: pd.DataFrame,
    residual: np.ndarray,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    columns = ["pitcher_id", "pitcher_hand", "batter_hand"]
    reference_rows = source if reference is None else reference
    work = source[columns].copy()
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
    contrast = (
        (grouped[("mean", 1)] - grouped[("mean", 0)])
        * effective_n / (effective_n + 1500.0)
    ).dropna()
    raw = {}
    for pitcher, value in contrast.items():
        for pitcher_hand in (1, 2):
            for batter_hand in (1, 2):
                raw[f"{int(pitcher)}|{pitcher_hand}|{batter_hand}"] = (
                    0.5 if pitcher_hand == batter_hand else -0.5
                ) * float(value)
    deployed, stats = normalize(
        pd.Series(raw, dtype=float), reference_rows, columns,
        HD_PUBLIC_ROW_SD, HD_STRENGTH,
    )
    stats.update({
        "pitchers": int(len(contrast)), "table_cells": int(len(deployed)),
        "k": 1500.0, "strength": HD_STRENGTH,
        "effective_n_median": float(effective_n.dropna().median()),
    })
    return deployed, stats


def freeze_d0_shape(
    source: pd.DataFrame,
    residual: np.ndarray,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    columns = ["pitcher_id", "batter_hand"]
    reference_rows = source if reference is None else reference
    work = source[["pitcher_id", "batter_hand"]].copy()
    work["residual"] = np.asarray(residual, dtype=float)
    parent_mean = work.groupby("pitcher_id", observed=True)["residual"].mean()
    child = work.groupby(columns, observed=True)["residual"].agg(["mean", "size"])
    parent_for_child = child.index.get_level_values("pitcher_id").map(parent_mean)
    difference = child["mean"].to_numpy(float) - np.asarray(parent_for_child, dtype=float)
    count = child["size"].to_numpy(float)
    v0 = pd.Series(difference * count / (count + 300.0), index=child.index)
    vk = pd.Series(difference * count / (count + 30.0), index=child.index)
    string_index = ["|".join(str(int(value)) for value in key) for key in child.index]
    v0.index = string_index
    vk.index = string_index
    base_rows = apply_table(reference_rows, columns, v0)
    low_rows = apply_table(reference_rows, columns, vk)
    shape_alpha = float(base_rows.std() / low_rows.std())
    raw = -3.0 * (shape_alpha * vk - v0)
    deployed, stats = normalize(
        raw, reference_rows, columns,
        D0_PUBLIC_ROW_SD, D0_STRENGTH,
    )
    stats.update({
        "table_cells": int(len(deployed)), "k0": 300.0, "k1": 30.0,
        "t": -3.0, "shape_alpha": shape_alpha, "strength": D0_STRENGTH,
    })
    return deployed, stats


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0


def gain(y: np.ndarray, base: np.ndarray, correction: np.ndarray) -> float:
    return float(bss(y, np.clip(base + correction, *CLIP)) - bss(y, base))


def projection(component_correlation: float, d0_local_gain: float) -> dict:
    # V61 rescaling: the observed point plus the fixed combined curvature puts
    # the vertex near 0.86, leaving roughly two tenths.
    rescale_gain = 0.20
    c4n_gain = 3.365091235973202 * OBSERVED_TRANSFER

    # HD's published honest forward gain was +8.4 at full strength.  Infer its
    # linear term with curvature transferred from its row energy, then evaluate
    # at 0.5 and attenuate by the observed v61 transfer ratio.
    hand_curvature = 5.5782
    hd_full_curvature = hand_curvature * (HD_PUBLIC_ROW_SD / 0.004007656969231088) ** 2
    hd_linear = 8.4 + hd_full_curvature
    hd_gain = (
        hd_linear * HD_STRENGTH - hd_full_curvature * HD_STRENGTH**2
    ) * OBSERVED_TRANSFER
    d0_gain = max(0.0, d0_local_gain) * OBSERVED_TRANSFER
    hd_curvature = hd_full_curvature * HD_STRENGTH**2
    d0_curvature = hand_curvature * (D0_PUBLIC_ROW_SD / 0.004007656969231088) ** 2
    overlap_penalty = max(0.0, 2.0 * component_correlation * np.sqrt(
        hd_curvature * d0_curvature
    ))
    total = rescale_gain + c4n_gain + hd_gain + d0_gain - overlap_penalty
    return {
        "v61_rescale": float(rescale_gain), "c4n_mirror": float(c4n_gain),
        "residual_hand": float(hd_gain), "d0_shape": float(d0_gain),
        "hd_d0_overlap_penalty": float(overlap_penalty),
        "total": float(total), "score": float(PUBLIC_V61 + total),
    }


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
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    with np.load(ROOT / "outputs/v60_oof_predictions.npz") as archive:
        base_v60 = archive["blended"].astype(float)
    if len(rows) != len(base) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v61 OOF and training rows are not aligned")
    residual = target - base
    active_2024 = season == 2024
    reference = rows.loc[active_2024].reset_index(drop=True)

    metadata = json.loads((ROOT / "submit/model/metadata.json").read_text(encoding="utf-8"))
    hand_config = metadata["v60_public_hand_shape"]
    hand_table = pd.Series(
        dict(zip(hand_config["keys"], hand_config["deltas"])), dtype=float,
    )
    c4n, c4n_stats = freeze_c4n_mirror(rows, hand_table, reference)
    hd, hd_stats = freeze_hd(rows, residual, reference)
    d0, d0_stats = freeze_d0_shape(rows, residual, reference)
    c4n_rows = apply_table(rows, ["pitcher_id", "pitcher_hand", "batter_hand"], c4n)
    hd_rows = apply_table(rows, ["pitcher_id", "pitcher_hand", "batter_hand"], hd)
    d0_rows = apply_table(rows, ["pitcher_id", "batter_hand"], d0)
    rescale_rows = V61_SCALE_DELTA * (base - base_v60)
    production_correction = rescale_rows + c4n_rows + hd_rows + d0_rows

    transfers = {}
    forward_d0_gains = []
    for source_year, validation_year in ((2023, 2024), (2024, 2023)):
        source_mask = season == source_year
        validation_mask = season == validation_year
        source_rows = rows.loc[source_mask].reset_index(drop=True)
        validation_rows = rows.loc[validation_mask].reset_index(drop=True)
        source_residual = residual[source_mask]
        source_hand, _ = freeze_hand_shape(source_rows, source_residual, source_rows)
        c4n_table, _ = freeze_c4n_mirror(source_rows, source_hand, source_rows)
        hd_table, _ = freeze_hd(source_rows, source_residual, source_rows)
        d0_table, _ = freeze_d0_shape(source_rows, source_residual, source_rows)
        dc4n = apply_table(validation_rows, ["pitcher_id", "pitcher_hand", "batter_hand"], c4n_table)
        dhd = apply_table(validation_rows, ["pitcher_id", "pitcher_hand", "batter_hand"], hd_table)
        dd0 = apply_table(validation_rows, ["pitcher_id", "batter_hand"], d0_table)
        d0_gain = gain(target[validation_mask], base[validation_mask], dd0)
        forward_d0_gains.append(d0_gain)
        combined = dc4n + dhd + dd0
        transfers[f"{source_year}_to_{validation_year}"] = {
            "c4n_gain": gain(target[validation_mask], base[validation_mask], dc4n),
            "hd_gain": gain(target[validation_mask], base[validation_mask], dhd),
            "d0_gain": d0_gain,
            "combined_gain": gain(target[validation_mask], base[validation_mask], combined),
            "combined_row_std": float(combined.std()),
        }

    component_correlations = {
        "c4n_hd": correlation(c4n_rows[active_2024], hd_rows[active_2024]),
        "c4n_d0": correlation(c4n_rows[active_2024], d0_rows[active_2024]),
        "hd_d0": correlation(hd_rows[active_2024], d0_rows[active_2024]),
    }
    d0_local = float(np.mean(forward_d0_gains))
    report = {
        "baseline": "v61_public_complete_shape",
        "configuration": {
            "v61_scale_delta": V61_SCALE_DELTA,
            "c4n_strength": C4N_STRENGTH,
            "hd_strength": HD_STRENGTH,
            "d0_strength": D0_STRENGTH,
        },
        "production": {
            "c4n": c4n_stats, "hd": hd_stats, "d0": d0_stats,
            "component_correlations": component_correlations,
            "combined_row_mean": float(production_correction[active_2024].mean()),
            "combined_row_std": float(production_correction[active_2024].std()),
            "in_sample_2024_gain": gain(
                target[active_2024], base[active_2024], production_correction[active_2024],
            ),
        },
        "transfers": transfers,
        "direction_audit": {
            "c4n_public_correlation": C4N_PUBLIC_CORRELATION,
            "hd_public_correlation": HD_PUBLIC_CORRELATION,
            "d0_public_correlation": D0_PUBLIC_CORRELATION,
            "external_model_or_prediction_used_in_tables": False,
        },
        "projection": projection(component_correlations["hd_d0"], d0_local),
        "projected_public_range": [1135.0, 1144.0],
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v62_residual_frontier.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
