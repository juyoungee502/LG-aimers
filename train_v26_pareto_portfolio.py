"""Freeze and validate the Pareto-robust v26 residual portfolio."""
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
from train_v25_temporal_portfolio import bss, regime_audit
from v25_temporal_portfolio import freeze_regime
from v26_pareto_policy import (
    FUTURES_CALIBRATION_POLICY, FUTURES_POLICY, REGULAR_POLICY,
)


VERSION = "v26_pareto_temporal_portfolio"
CLIP = (.005, .995)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def main():
    root = Path(__file__).resolve().parent
    model_dir = root / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    accepted = {
        "v24_robust_command_resolution",
        "v25_strict_temporal_portfolio",
        VERSION,
    }
    if metadata.get("version") not in accepted:
        raise ValueError(
            f"Expected v24/v25/v26 artifacts, got {metadata.get('version')}"
        )

    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(raw, target_series)
    features_all = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
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
    if min_regular < 4.9 or min_futures < 10.0:
        raise RuntimeError(
            f"v26 Pareto promotion gate failed: R={min_regular}, F={min_futures}"
        )

    upgraded = base.copy()
    upgraded[regular_valid] = regular_prediction
    upgraded[futures_valid] = futures_prediction
    active_2024 = year == 2024
    v24_score = bss(y[active_2024], base[active_2024])
    v26_score = bss(y[active_2024], upgraded[active_2024])
    report = {
        "v24_2024_bss": v24_score,
        "v26_2024_bss": v26_score,
        "gain_2024": v26_score - v24_score,
        "regular_gain_2024": regular_metrics["2024/all"],
        "futures_gain_2024": futures_metrics["2024/all"],
        "minimum_regular_segment_gain": min_regular,
        "minimum_futures_segment_gain": min_futures,
        "regular_metrics": regular_metrics,
        "futures_metrics": futures_metrics,
    }
    np.savez_compressed(
        root / "outputs/v26_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )

    latest = year == 2024
    deploy = {}
    for regime, key, policy, calibration in (
        ("R", "regular", REGULAR_POLICY, ()),
        ("F", "futures", FUTURES_POLICY, FUTURES_CALIBRATION_POLICY),
    ):
        source = np.flatnonzero(
            latest & rows["game_type"].eq(regime).to_numpy()
        )
        deploy[key] = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], y[source],
            policy, calibration,
        )
    deploy.update({
        "source_season": 2024,
        "target_season": 2025,
        "game_type_regular": "R",
        "game_type_futures": "F",
        "row_independent_inference": True,
        "base_prediction_version": "v24_robust_command_resolution",
    })

    metadata["version"] = VERSION
    names = metadata.setdefault("model_names", [])
    if "v26_pareto_portfolio" not in names:
        names.append("v26_pareto_portfolio")
    metadata["v26_pareto_portfolio"] = deploy
    metadata.setdefault("training_info", {})["v26_validation"] = {
        **report,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print("v26 validation:", json.dumps(report), flush=True)
    print("Stored v26 frozen tables, metadata, and OOF diagnostics", flush=True)


if __name__ == "__main__":
    main()
