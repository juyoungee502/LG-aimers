"""Upgrade v12 artifacts using recent-season residual-table geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from residual_effects import apply_residual_effects, build_residual_effects
from residual_geometry import apply_v13_geometry


TARGET = "control_success"
SOURCE_COLUMNS = [
    "season", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "balls_before", "strikes_before", "num_runners_on", TARGET,
]
MODEL_INDICES = [2, 4, 5, 6, 7]


def score(target, prediction):
    rate = float(np.mean(target))
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def fit_source_blends(target, years, matrix, gate, source_year):
    parameters = {}
    for label, gate_value in (("other", False), ("two_strike", True)):
        mask = (years == source_year) & (gate == gate_value)
        y = target[mask]
        x = matrix[mask][:, MODEL_INDICES]
        reference = float(y.mean() * (1. - y.mean()))
        count = x.shape[1]

        def objective(z):
            prediction = np.clip(z[count] + z[count + 1] * (x @ z[:count]), .005, .995)
            return float(
                np.mean((y - prediction) ** 2) / reference
                + .005 * z[count] ** 2 + .002 * (z[count + 1] - 1.) ** 2
            )

        result = minimize(
            objective, np.r_[np.full(count, 1. / count), 0., 1.], method="SLSQP",
            bounds=[(0., 1.)] * count + [(-.08, .08), (.75, 1.25)],
            constraints={"type": "eq", "fun": lambda z: z[:count].sum() - 1.},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"Source blend optimization failed: {result.message}")
        parameters[label] = result.x
    return parameters


def apply_blends(matrix, gate, parameters):
    candidates = matrix[:, MODEL_INDICES]
    prediction = np.empty(len(matrix), dtype=np.float64)
    for label, gate_value in (("other", False), ("two_strike", True)):
        mask = gate == gate_value
        values = parameters[label]
        prediction[mask] = np.clip(
            values[-2] + values[-1] * (candidates[mask] @ values[:-2]), .005, .995
        )
    return prediction


def main():
    root = Path(__file__).resolve().parent
    metadata_path = root / "submit" / "model" / "metadata.json"
    diagnostic_path = root / "outputs" / "v12_oof_predictions.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v12_transferred_residual_effects":
        raise ValueError(f"Expected v12 artifacts, found {metadata.get('version')}")

    with np.load(diagnostic_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    raw = pd.read_csv(root / "data" / "train.csv", usecols=SOURCE_COLUMNS, encoding="utf-8-sig")
    rows = pd.concat([raw.loc[raw.season.eq(year)] for year in (2023, 2024)], ignore_index=True)
    target = data["target"].astype(np.float64)
    years = data["season"]
    matrix = data["predictions"].astype(np.float64)
    gate = data["two_strike"].astype(bool)
    if len(rows) != len(target) or not np.allclose(rows[TARGET].to_numpy(), target):
        raise ValueError("OOF rows do not align with train.csv")

    source, latest = years == 2023, years == 2024
    source_parameters = fit_source_blends(target, years, matrix, gate, 2023)
    honest_base = apply_blends(matrix, gate, source_parameters)
    source_rows = rows.loc[source].reset_index(drop=True)
    latest_rows = rows.loc[latest].reset_index(drop=True)
    source_residual = target[source] - honest_base[source]
    v12_validation_effects = build_residual_effects(source_rows, source_residual)
    v12_adjustment, _ = apply_residual_effects(latest_rows, v12_validation_effects)
    v13_validation_effects = apply_v13_geometry(
        source_rows, source_residual, v12_validation_effects
    )
    v13_adjustment, _ = apply_residual_effects(latest_rows, v13_validation_effects)
    v12_prediction = np.clip(honest_base[latest] + v12_adjustment, .005, .995)
    v13_prediction = np.clip(honest_base[latest] + v13_adjustment, .005, .995)
    base_score = score(target[latest], honest_base[latest])
    v12_score = score(target[latest], v12_prediction)
    v13_score = score(target[latest], v13_prediction)
    print(
        f"Honest 2023->2024 BSS: base={base_score:.4f}; "
        f"v12={v12_score:.4f}; v13={v13_score:.4f}; "
        f"v13-v12={v13_score-v12_score:+.4f}"
    )

    # Production uses only the most recent strictly OOF residual season. This
    # beat a two-season source window in each of the three latest transfers.
    production_base = data["base_blended"].astype(np.float64)
    production_rows = rows.loc[latest].reset_index(drop=True)
    production_residual = target[latest] - production_base[latest]
    final_effects = build_residual_effects(production_rows, production_residual)
    final_effects = apply_v13_geometry(production_rows, production_residual, final_effects)

    metadata["version"] = "v13_recent_residual_geometry"
    metadata["residual_effects"] = final_effects
    metadata["training_info"]["v13_validation"] = {
        "source_year": 2023,
        "target_year": 2024,
        "base_bss": base_score,
        "v12_bss": v12_score,
        "v13_bss": v13_score,
        "production_residual_years": [2024],
        "rolling_geometry_delta": {"2021": 34.818, "2022": -26.406, "2023": 6.329, "2024": 6.656},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    diagnostic_prediction = honest_base.copy()
    diagnostic_prediction[latest] = v13_prediction
    np.savez_compressed(
        root / "outputs" / "v13_oof_predictions.npz",
        predictions=matrix, target=target.astype(np.float32), season=years,
        model_names=data["model_names"], two_strike=gate,
        base_blended=honest_base, blended=diagnostic_prediction,
    )
    print(f"Upgraded artifacts: {metadata_path}")


if __name__ == "__main__":
    main()
