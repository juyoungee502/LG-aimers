"""Audit regular-season ensemble weights with a strict 2023 -> 2024 transfer.

This is a research-only script.  It never writes production model artifacts.
The final 2025 weights are reported only after the same fitting rule improves the
next unseen season (fit 2023, evaluate 2024).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "outputs/v24_oof_predictions.npz"
MODEL_DIR = ROOT / "submit/model"
RIDGES = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    rate = float(target.mean())
    return 100000.0 * (1.0 - np.mean((prediction - target) ** 2) / (rate * (1.0 - rate)))


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.005, 0.995)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def fit_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    intercept: float,
    slope: float,
    ridge: float,
) -> np.ndarray:
    """Fit simplex weights while freezing the already-deployed calibration."""
    reference = float(target.mean() * (1.0 - target.mean()))

    def objective(weights: np.ndarray) -> float:
        prediction = np.clip(intercept + slope * (matrix @ weights), 0.005, 0.995)
        loss = np.mean((prediction - target) ** 2) / reference
        return float(loss + ridge * np.sum((weights - prior) ** 2))

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(prior),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def apply_weights(
    matrix: np.ndarray,
    gate: np.ndarray,
    parameters: dict[str, dict[str, np.ndarray | float]],
) -> np.ndarray:
    result = np.empty(len(matrix), dtype=float)
    for label, value in (("other", False), ("two_strike", True)):
        active = gate == value
        parameter = parameters[label]
        result[active] = np.clip(
            float(parameter["intercept"])
            + float(parameter["slope"]) * (matrix[active] @ parameter["weights"]),
            0.005,
            0.995,
        )
    return result


def segments(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    month = rows["game_month"].to_numpy()
    size = len(rows)
    return {
        "all": np.ones(size, dtype=bool),
        "half_1": np.arange(size) < size // 2,
        "half_2": np.arange(size) >= size // 2,
        "q1": np.arange(size) < size // 4,
        "q2": (np.arange(size) >= size // 4) & (np.arange(size) < size // 2),
        "q3": (np.arange(size) >= size // 2) & (np.arange(size) < 3 * size // 4),
        "q4": np.arange(size) >= 3 * size // 4,
        **{f"month_{value}": month == value for value in np.unique(month)},
    }


def gain_report(target: np.ndarray, base: np.ndarray, candidate: np.ndarray, rows: pd.DataFrame):
    return {
        label: bss(target[active], candidate[active]) - bss(target[active], base[active])
        for label, active in segments(rows).items() if active.any()
    }


def main() -> None:
    with np.load(ARCHIVE) as archive:
        values = {key: archive[key] for key in archive.files}
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    names = values["model_names"].astype(str).tolist()
    deployed = metadata["segment_blends"]
    candidate_names = list(deployed["other"]["weights"])
    indices = [names.index(name) for name in candidate_names]
    matrix = values["predictions"][:, indices].astype(float)
    target = values["target"].astype(float)
    season = values["season"].astype(int)
    gate = values["two_strike"].astype(bool)
    base_blended = values["base_blended"].astype(float)
    final_blended = values["blended"].astype(float)

    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["season", "game_type", "game_month", "control_success"],
        encoding="utf-8-sig",
    )
    rows = pd.concat(
        [raw.loc[raw["season"].eq(value)] for value in sorted(np.unique(season))],
        ignore_index=True,
    )
    if len(rows) != len(target) or not np.array_equal(rows["season"].to_numpy(), season):
        raise ValueError("OOF archive and train.csv are not aligned")
    if not np.allclose(rows["control_success"].to_numpy(), target):
        raise ValueError("OOF targets and train.csv are not aligned")

    regular = rows["game_type"].eq("R").to_numpy()
    source = regular & (season == 2023)
    future = regular & (season == 2024)
    print("candidate_names", candidate_names)
    print("deployed base R", {
        str(year): bss(target[regular & (season == year)], base_blended[regular & (season == year)])
        for year in (2023, 2024)
    })
    results = []
    for ridge in RIDGES:
        parameters = {}
        for label, value in (("other", False), ("two_strike", True)):
            parameter = deployed[label]
            prior = np.asarray([parameter["weights"][name] for name in candidate_names])
            active = source & (gate == value)
            weights = fit_weights(
                matrix[active], target[active], prior,
                float(parameter["intercept"]), float(parameter["slope"]), ridge,
            )
            parameters[label] = {
                "weights": weights,
                "intercept": float(parameter["intercept"]),
                "slope": float(parameter["slope"]),
            }
        alternative_base = apply_weights(matrix, gate, parameters)
        # Keep F untouched.  Audit both probability-space and logit-space replacement
        # because later v16-v24 residuals were applied after the deployed base blend.
        probability_candidate = np.clip(
            final_blended + regular * (alternative_base - base_blended), 0.005, 0.995,
        )
        logit_candidate = sigmoid(
            logit(final_blended) + regular * (logit(alternative_base) - logit(base_blended))
        )
        direct_gain = bss(target[future], alternative_base[future]) - bss(target[future], base_blended[future])
        probability_gains = gain_report(
            target[future], final_blended[future], probability_candidate[future],
            rows.loc[future].reset_index(drop=True),
        )
        logit_gains = gain_report(
            target[future], final_blended[future], logit_candidate[future],
            rows.loc[future].reset_index(drop=True),
        )
        result = {
            "ridge": ridge,
            "fit_gain_2023_base": bss(target[source], alternative_base[source]) - bss(target[source], base_blended[source]),
            "transfer_gain_2024_base": direct_gain,
            "probability_gains": probability_gains,
            "logit_gains": logit_gains,
            "parameters": {label: parameters[label]["weights"].tolist() for label in parameters},
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = ROOT / "research/v24_forward_ensemble.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", output)


if __name__ == "__main__":
    main()
