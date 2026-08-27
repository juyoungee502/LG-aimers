"""Jointly audit a forward-fitted base reblend as an addition to v26.

For every chronological transfer, ensemble weights are fitted only on the
source block.  The resulting change in the pre-residual base is then added to
the independently source-frozen v26 correction and evaluated on the unseen
block.  This tests whether the strong standalone reblend contains information
that survives the existing downstream stack.
"""
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
from research_v24_forward_ensemble import apply_weights, fit_weights
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime
from v26_pareto_policy import REGULAR_POLICY


ROOT = Path(__file__).resolve().parent
RIDGES = (0., 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3, 1.)
WEIGHTS = np.round(np.arange(-.5, .501, .01), 2)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def fitted_reblend(
    matrix: np.ndarray, target: np.ndarray, gate: np.ndarray,
    source: np.ndarray, deployed: dict, candidate_names: list[str], ridge: float,
) -> dict:
    parameters = {}
    for label, value in (("other", False), ("two_strike", True)):
        parameter = deployed[label]
        prior = np.asarray([
            parameter["weights"][name] for name in candidate_names
        ])
        active = source[gate[source] == value]
        parameters[label] = {
            "weights": fit_weights(
                matrix[active], target[active], prior,
                float(parameter["intercept"]), float(parameter["slope"]), ridge,
            ),
            "intercept": float(parameter["intercept"]),
            "slope": float(parameter["slope"]),
        }
    return parameters


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
    metadata = json.loads((
        ROOT / "submit/model/metadata.json"
    ).read_text(encoding="utf-8"))
    deployed = metadata["segment_blends"]
    candidate_names = list(deployed["other"]["weights"])
    model_names = oof["model_names"].astype(str).tolist()
    model_indices = [model_names.index(name) for name in candidate_names]

    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == value) for value in (2023, 2024)
    ])
    rows = raw.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    year = oof["season"].astype(int)
    base = oof["blended"].astype(float)
    pre_residual = oof["base_blended"].astype(float)
    matrix = oof["predictions"][:, model_indices].astype(float)
    gate = oof["two_strike"].astype(bool)
    if not np.allclose(target_all[positions], y):
        raise ValueError("v24 OOF rows do not align with train.csv")

    regular = rows["game_type"].eq("R").to_numpy()
    indices = {
        value: np.flatnonzero(regular & (year == value)) for value in (2023, 2024)
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

    output = []
    for ridge in RIDGES:
        blocks = []
        for label, source, valid in transfers:
            v26 = freeze_regime(
                rows.iloc[source], features.iloc[source], base[source], y[source],
                REGULAR_POLICY, (),
            )
            correction = apply_regime(
                rows.iloc[valid], features.iloc[valid], base[valid], v26,
            )
            parameters = fitted_reblend(
                matrix, y, gate, source, deployed, candidate_names, ridge,
            )
            alternative = apply_weights(matrix[valid], gate[valid], parameters)
            delta = alternative - pre_residual[valid]
            blocks.append({
                "label": label, "valid": valid, "correction": correction,
                "delta": delta,
                "masks": segment_masks(
                    rows.iloc[valid].reset_index(drop=True), label,
                ),
            })

        for weight in WEIGHTS:
            metrics = {}
            for block in blocks:
                valid = block["valid"]
                candidate = np.clip(
                    base[valid] + block["correction"]
                    + float(weight) * block["delta"], .005, .995,
                )
                for name, active in block["masks"].items():
                    metrics[name] = bss(
                        y[valid][active], candidate[active],
                    ) - bss(y[valid][active], base[valid][active])
            output.append({
                "ridge": ridge,
                "weight": float(weight),
                "gain_2024": metrics["2024/all"],
                "minimum_strict_gain": min(metrics.values()),
                "metrics": metrics,
            })

    safe = sorted(
        (item for item in output if item["minimum_strict_gain"] >= 0.),
        key=lambda item: (item["gain_2024"], item["minimum_strict_gain"]),
        reverse=True,
    )
    floor_five = sorted(
        (item for item in output if item["minimum_strict_gain"] >= 5.),
        key=lambda item: (item["gain_2024"], item["minimum_strict_gain"]),
        reverse=True,
    )
    robust = sorted(
        output,
        key=lambda item: (item["minimum_strict_gain"], item["gain_2024"]),
        reverse=True,
    )
    report = {"safe": safe[:30], "floor_five": floor_five[:30], "robust": robust[:30]}
    path = ROOT / "research/v27_base_reblend.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: [
            {field: item[field] for field in (
                "ridge", "weight", "gain_2024", "minimum_strict_gain",
            )}
            for item in values[:10]
        ]
        for key, values in report.items()
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
