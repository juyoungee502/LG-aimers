"""Audit global R/F strength multipliers around the frozen v26 policy."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime
from v26_pareto_policy import (
    FUTURES_CALIBRATION_POLICY, FUTURES_POLICY, REGULAR_POLICY,
)


ROOT = Path(__file__).resolve().parent
SCALES = np.round(np.arange(.5, 2.001, .025), 3)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def audit(rows, features, y, base, year, regime, policy, calibration):
    active = rows["game_type"].eq(regime).to_numpy()
    indices = {
        value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        ("2024", indices[2023], indices[2024]),
    )
    blocks = []
    for label, source, valid in transfers:
        frozen = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], y[source],
            policy, calibration,
        )
        correction = apply_regime(
            rows.iloc[valid], features.iloc[valid], base[valid], frozen,
        )
        blocks.append((
            label, valid, correction,
            segment_masks(rows.iloc[valid].reset_index(drop=True), label),
        ))
    reports = []
    for scale in SCALES:
        metrics = {}
        for _label, valid, correction, masks in blocks:
            candidate = np.clip(base[valid] + scale * correction, .005, .995)
            for name, mask in masks.items():
                metrics[name] = (
                    bss(y[valid][mask], candidate[mask])
                    - bss(y[valid][mask], base[valid][mask])
                )
        reports.append({
            "scale": float(scale), "gain_2024": metrics["2024/all"],
            "minimum_strict_gain": min(metrics.values()), "metrics": metrics,
        })
    return sorted(reports, key=lambda item: item["gain_2024"], reverse=True)


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
        "R": audit(rows, features, y, base, year, "R", REGULAR_POLICY, ()),
        "F": audit(
            rows, features, y, base, year, "F", FUTURES_POLICY,
            FUTURES_CALIBRATION_POLICY,
        ),
    }
    path = ROOT / "research/v27_policy_scale.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {}
    for regime, reports in result.items():
        summary[regime] = {
            "best": reports[:5],
            "safe": sorted(
                (item for item in reports if item["minimum_strict_gain"] >= 0.),
                key=lambda item: item["gain_2024"], reverse=True,
            )[:5],
            "floor_five": sorted(
                (item for item in reports if item["minimum_strict_gain"] >= 5.),
                key=lambda item: item["gain_2024"], reverse=True,
            )[:5],
        }
        for key in summary[regime]:
            summary[regime][key] = [
                {field: item[field] for field in (
                    "scale", "gain_2024", "minimum_strict_gain",
                )}
                for item in summary[regime][key]
            ]
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
