"""Compare deployed v57 correction magnitude with its forward-fold proxy."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v40_failure_seed_stability import logit, sigmoid
from v25_temporal_portfolio import apply_regime


ROOT = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=PerformanceWarning)


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    changed = values != 0.0
    return {
        "rows": int(len(values)),
        "changed_rows": int(changed.sum()),
        "mean_abs_all": float(np.mean(np.abs(values))),
        "mean_abs_changed": float(np.mean(np.abs(values[changed]))),
        "mean_square_all": float(np.mean(values ** 2)),
        "max_abs": float(np.max(np.abs(values))),
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    history = training_history_arrays(raw, target_series)
    features_all = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features_all, raw)
    features_all = add_state_interactions(features_all)

    positions = np.concatenate([
        np.flatnonzero(raw["season"].to_numpy() == year) for year in (2023, 2024)
    ])
    rows = raw.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v57_oof_predictions.npz") as archive:
        v57 = {key: archive[key] for key in archive.files}

    year = v38["season"].astype(int)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    active = (year == 2024) & regular
    v56 = v54["blended"].astype(float).copy()
    active_f = (year == 2024) & ~regular
    v56[active_f] = sigmoid(
        logit(v38["blended"][active_f].astype(float))
        + 1.25 * (
            logit(v54["blended"][active_f].astype(float))
            - logit(v38["blended"][active_f].astype(float))
        )
    )
    validation_correction = (
        v57["blended"][active].astype(float) - v56[active]
    )

    metadata = json.loads(
        (ROOT / "submit/model/metadata.json").read_text(encoding="utf-8")
    )
    configuration = metadata["v57_conservative_r_table"]
    deploy_correction = apply_regime(
        rows.loc[active], features.loc[active], v56[active],
        configuration["regular"],
    )
    gate_config = configuration["exposure_gate"]
    gate = (
        features.loc[active, gate_config["feature"]].to_numpy(float)
        > float(gate_config["minimum_exclusive"])
    )
    deploy_correction *= gate.astype(float)

    validation_summary = summary(validation_correction)
    deploy_summary = summary(deploy_correction)
    report = {
        "validation_forward_table": validation_summary,
        "production_2024_table_on_2024_feature_distribution": deploy_summary,
        "mean_square_ratio_deploy_to_validation": float(
            deploy_summary["mean_square_all"]
            / validation_summary["mean_square_all"]
        ),
        "interpretation": (
            "Magnitude-only comparison; production-table target correlation is "
            "not scored on its source season."
        ),
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v58_deploy_curvature.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
