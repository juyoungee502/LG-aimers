"""Test predeclared season-exposure gates for the single v57 R table."""
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
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime


ROOT = Path(__file__).resolve().parent
F_SCALE = 1.25
POLICY = ({
    "type": "one_d", "kind": "numeric",
    "column": "pitcher_success_x_runners",
    "bins": 8, "shrink": 6400.0, "scale": 1.0, "weight": 0.5,
},)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def gate_values(season_n: np.ndarray) -> dict[str, np.ndarray]:
    season_n = np.asarray(season_n, dtype=float)
    result = {"none": np.ones(len(season_n), dtype=float)}
    for threshold in (25.0, 50.0, 100.0):
        result[f"hard_{int(threshold)}"] = (season_n > threshold).astype(float)
    for threshold in (25.0, 50.0, 100.0, 200.0):
        result[f"ramp_{int(threshold)}"] = np.clip(
            season_n / threshold, 0.0, 1.0,
        )
    result["delayed_ramp_25_100"] = np.clip(
        (season_n - 25.0) / 75.0, 0.0, 1.0,
    )
    return result


def main() -> None:
    full = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = full.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(full, target_series)
    features_all = engineer_features(
        full, *history, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features_all, full)
    features_all = add_state_interactions(features_all)
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}

    seasons = full["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    rows = full.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    target = v38["target"].astype(float)
    year = v38["season"].astype(int)
    if not np.allclose(target_all[positions], target):
        raise ValueError("OOF rows do not align")
    base = v54["blended"].astype(float).copy()
    active_2024_f = (
        (year == 2024) & rows["game_type"].astype(str).eq("F").to_numpy()
    )
    base[active_2024_f] = sigmoid(
        logit(v38["blended"][active_2024_f].astype(float))
        + F_SCALE * (
            logit(v54["blended"][active_2024_f].astype(float))
            - logit(v38["blended"][active_2024_f].astype(float))
        )
    )

    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    indices = {
        value: np.flatnonzero(regular & (year == value))
        for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index) // 2] if half == 1 else index[len(index) // 2:]
        for value, index in indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        ("2024", indices[2023], indices[2024]),
    )
    metrics: dict[str, dict[str, float]] = {
        name: {} for name in gate_values(np.ones(1))
    }
    for label, source, valid in transfers:
        frozen = freeze_regime(
            rows.iloc[source], features.iloc[source], base[source], target[source],
            POLICY, (),
        )
        direction = apply_regime(
            rows.iloc[valid], features.iloc[valid], base[valid], frozen,
        )
        gates = gate_values(features.iloc[valid]["pitcher_season_n"].to_numpy(float))
        query = rows.iloc[valid].reset_index(drop=True)
        for gate_name, gate in gates.items():
            candidate = np.clip(base[valid] + gate * direction, 0.005, 0.995)
            for segment, active in segment_masks(query, label).items():
                metrics[gate_name][segment] = float(
                    bss(target[valid][active], candidate[active])
                    - bss(target[valid][active], base[valid][active])
                )

    reports = []
    for name, values in metrics.items():
        reports.append({
            "gate": name,
            "gain_2024_R": values["2024/all"],
            "min_transfer_all": min(
                values[f"{label}/all"] for label in (
                    "23h1_to_23h2", "23_to_24h1", "23_to_24h2",
                    "24h1_to_24h2",
                )
            ),
            "min_quarter": min(value for key, value in values.items() if "/q" in key),
            "min_half": min(value for key, value in values.items() if "/half_" in key),
            "min_month": min(value for key, value in values.items() if "/month_" in key),
            "metrics": values,
        })
    ranked = sorted(reports, key=lambda row: (
        min(row["min_transfer_all"], row["min_quarter"], row["min_half"]),
        row["gain_2024_R"], row["min_month"],
    ), reverse=True)
    output = {
        "feature": "pitcher_season_n",
        "predeclared_gates": [row["gate"] for row in reports],
        "ranked": ranked,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v57_exposure_gate.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(ranked, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
