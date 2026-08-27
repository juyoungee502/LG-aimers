"""Freeze and validate the strict chronological v25 residual portfolio."""
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
from v25_temporal_portfolio import (
    FUTURES_CALIBRATION_POLICY, FUTURES_POLICY, REGULAR_POLICY,
    apply_regime, freeze_regime,
)


VERSION = "v25_strict_temporal_portfolio"
CLIP = (.005, .995)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    reference = float(target.mean() * (1. - target.mean()))
    return float(100000. * (1. - np.mean((target - prediction) ** 2) / reference))


def segment_masks(rows: pd.DataFrame, prefix: str):
    position = np.arange(len(rows))
    result = {
        f"{prefix}/all": np.ones(len(rows), dtype=bool),
        f"{prefix}/half_1": position < len(rows) // 2,
        f"{prefix}/half_2": position >= len(rows) // 2,
        f"{prefix}/q1": position < len(rows) // 4,
        f"{prefix}/q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        f"{prefix}/q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        f"{prefix}/q4": position >= 3 * len(rows) // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        if active.sum() >= 40:
            result[f"{prefix}/month_{int(month)}"] = active
    return result


def regime_audit(rows, features, y, base, year, regime, policy, calibration):
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
    )
    metrics = {}
    for label, source, valid in transfers:
        frozen = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], y[source],
            policy, calibration,
        )
        candidate = np.clip(
            base[valid] + apply_regime(
                rows.iloc[valid], features.iloc[valid], base[valid], frozen,
            ), *CLIP,
        )
        query = rows.iloc[valid].reset_index(drop=True)
        for name, mask in segment_masks(query, label).items():
            metrics[name] = bss(y[valid][mask], candidate[mask]) - bss(
                y[valid][mask], base[valid][mask],
            )
    source, valid = indices[2023], indices[2024]
    frozen = freeze_regime(
        rows.iloc[source], features.iloc[source], base[source], y[source],
        policy, calibration,
    )
    candidate = np.clip(
        base[valid] + apply_regime(
            rows.iloc[valid], features.iloc[valid], base[valid], frozen,
        ), *CLIP,
    )
    query = rows.iloc[valid].reset_index(drop=True)
    for name, mask in segment_masks(query, "2024").items():
        metrics[name] = bss(y[valid][mask], candidate[mask]) - bss(
            y[valid][mask], base[valid][mask],
        )
    return metrics, valid, candidate


def main():
    root = Path(__file__).resolve().parent
    model_dir = root / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v24_robust_command_resolution", VERSION,
    ):
        raise ValueError(f"Expected v24/v25 artifacts, got {metadata.get('version')}")

    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    bases = training_history_arrays(raw, target_series)
    features_all = engineer_features(
        raw, *bases, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features_all, raw)
    features_all = add_state_interactions(features_all)

    with np.load(root / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == value) for value in (2023, 2024)
    ])
    rows = raw.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    if not np.allclose(target_all[positions], y):
        raise ValueError("v24 OOF rows do not align with train.csv")

    regular_metrics, regular_valid, regular_prediction = regime_audit(
        rows, features, y, base, year, "R", REGULAR_POLICY, (),
    )
    futures_metrics, futures_valid, futures_prediction = regime_audit(
        rows, features, y, base, year, "F", FUTURES_POLICY,
        FUTURES_CALIBRATION_POLICY,
    )
    min_regular = min(regular_metrics.values())
    min_futures = min(futures_metrics.values())
    if min_regular < 7.0 or min_futures < 20.0:
        raise RuntimeError(
            f"v25 strict promotion gate failed: R={min_regular}, F={min_futures}"
        )

    upgraded = base.copy()
    upgraded[regular_valid] = regular_prediction
    upgraded[futures_valid] = futures_prediction
    active_2024 = year == 2024
    report = {
        "v24_2024_bss": bss(y[active_2024], base[active_2024]),
        "v25_2024_bss": bss(y[active_2024], upgraded[active_2024]),
        "gain_2024": bss(y[active_2024], upgraded[active_2024])
        - bss(y[active_2024], base[active_2024]),
        "regular_gain_2024": regular_metrics["2024/all"],
        "futures_gain_2024": futures_metrics["2024/all"],
        "minimum_regular_segment_gain": min_regular,
        "minimum_futures_segment_gain": min_futures,
        "regular_metrics": regular_metrics,
        "futures_metrics": futures_metrics,
    }
    np.savez_compressed(
        root / "outputs/v25_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )

    latest = year == 2024
    deploy = {}
    for regime, key, policy, calibration in (
        ("R", "regular", REGULAR_POLICY, ()),
        ("F", "futures", FUTURES_POLICY, FUTURES_CALIBRATION_POLICY),
    ):
        source = np.flatnonzero(latest & rows["game_type"].eq(regime).to_numpy())
        deploy[key] = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], y[source],
            policy, calibration,
        )
    deploy.update({
        "source_season": 2024, "target_season": 2025,
        "game_type_regular": "R", "game_type_futures": "F",
        "row_independent_inference": True,
    })
    metadata["version"] = VERSION
    names = metadata.setdefault("model_names", [])
    if "v25_temporal_portfolio" not in names:
        names.append("v25_temporal_portfolio")
    metadata["v25_temporal_portfolio"] = deploy
    metadata["training_info"]["v25_validation"] = {
        **report, "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print("v25 validation:", json.dumps(report), flush=True)
    print("Stored v25 frozen tables, metadata, and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
