"""Add recency-weighted native-categorical count specialists to v14."""
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
CANDIDATE_NAMES = [
    "catboost", "count_expert", "categorical_catboost",
    "categorical_count_expert", "brier_regressor", "weighted_catboost",
    "weighted_categorical_specialist",
]
BASE_NAMES = CANDIDATE_NAMES[:-1]


def score(target, prediction):
    rate = float(np.mean(target))
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def fit_blends(target, years, candidates, gate, source_year):
    parameters = {}
    count = candidates.shape[1]
    for label, gate_value in (("other", False), ("two_strike", True)):
        mask = (years == source_year) & (gate == gate_value)
        y, x = target[mask], candidates[mask]
        reference = float(y.mean() * (1. - y.mean()))

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
            raise RuntimeError(f"Blend optimization failed: {result.message}")
        parameters[label] = result.x
    return parameters


def apply_blends(candidates, gate, parameters):
    prediction = np.empty(len(candidates), dtype=np.float64)
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
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v14_weighted_catboost":
        raise ValueError(f"Expected v14 artifacts, found {metadata.get('version')}")
    with np.load(root / "outputs" / "v14_oof_predictions.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    with np.load(
        root / "outputs" / "v15_weighted_categorical_oof_predictions.npz",
        allow_pickle=False,
    ) as loaded:
        specialist = {key: loaded[key] for key in loaded.files}

    target = data["target"].astype(np.float64)
    years = data["season"]
    gate = data["two_strike"].astype(bool)
    if not np.array_equal(years, specialist["season"]) or not np.allclose(
        target, specialist["target"]
    ):
        raise ValueError("Specialist OOF rows do not align with v14 diagnostics")
    raw_names = list(data["model_names"].astype(str))
    indices = [raw_names.index(name) for name in BASE_NAMES]
    base_matrix = data["predictions"].astype(np.float64)
    candidates = np.column_stack([
        base_matrix[:, indices], specialist["prediction"].astype(np.float64),
    ])

    raw = pd.read_csv(root / "data" / "train.csv", usecols=SOURCE_COLUMNS, encoding="utf-8-sig")
    rows = pd.concat([raw.loc[raw.season.eq(year)] for year in (2023, 2024)], ignore_index=True)
    if len(rows) != len(target) or not np.allclose(rows[TARGET].to_numpy(), target):
        raise ValueError("OOF rows do not align with train.csv")
    source, latest = years == 2023, years == 2024

    honest_parameters = fit_blends(target, years, candidates, gate, 2023)
    honest_base = apply_blends(candidates, gate, honest_parameters)
    source_rows = rows.loc[source].reset_index(drop=True)
    latest_rows = rows.loc[latest].reset_index(drop=True)
    validation_effects = build_residual_effects(
        source_rows, target[source] - honest_base[source]
    )
    validation_effects = apply_v13_geometry(
        source_rows, target[source] - honest_base[source], validation_effects
    )
    validation_adjustment, _ = apply_residual_effects(latest_rows, validation_effects)
    honest_prediction = np.clip(
        honest_base[latest] + validation_adjustment, .005, .995
    )

    production_parameters = fit_blends(target, years, candidates, gate, 2024)
    production_base = apply_blends(candidates, gate, production_parameters)
    production_rows = rows.loc[latest].reset_index(drop=True)
    production_residual = target[latest] - production_base[latest]
    final_effects = build_residual_effects(production_rows, production_residual)
    final_effects = apply_v13_geometry(
        production_rows, production_residual, final_effects
    )

    metadata["version"] = "v15_weighted_categorical_specialist"
    metadata["model_names"] = raw_names + ["weighted_categorical_specialist"]
    metadata["segment_blends"] = {
        label: {
            "weights": dict(zip(CANDIDATE_NAMES, values[:-2].tolist())),
            "intercept": float(values[-2]), "slope": float(values[-1]),
        }
        for label, values in production_parameters.items()
    }
    metadata["residual_effects"] = final_effects
    metadata["training_info"]["v15_validation"] = {
        "honest_2023_to_2024_bss": score(target[latest], honest_prediction),
        "production_base_2024_bss": score(target[latest], production_base[latest]),
        "research_five_block_v14_bss": 906.2197,
        "research_five_block_v15_bss": 909.4044,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    diagnostic_prediction = honest_base.copy()
    diagnostic_prediction[latest] = honest_prediction
    np.savez_compressed(
        root / "outputs" / "v15_oof_predictions.npz",
        predictions=np.column_stack([base_matrix, specialist["prediction"]]),
        target=target.astype(np.float32), season=years,
        model_names=np.asarray(metadata["model_names"]), two_strike=gate,
        base_blended=honest_base, blended=diagnostic_prediction,
    )
    print(
        f"v15 honest 2023->2024 BSS={score(target[latest], honest_prediction):.4f}; "
        f"production base 2024 BSS={score(target[latest], production_base[latest]):.4f}"
    )
    print(
        "Production blends:",
        {key: np.round(value, 6).tolist() for key, value in production_parameters.items()},
    )


if __name__ == "__main__":
    main()
