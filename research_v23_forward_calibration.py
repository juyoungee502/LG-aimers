"""Forward-audit fixed, row-independent calibration policies over v23.

Parameters are fitted on 2023 predictions/targets and transferred unchanged to
2024.  No statistic from the validation batch is used to form its prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def best_affine(target, prediction, active=None):
    if active is None:
        active = np.ones(len(target), dtype=bool)
    y = target[active]
    z = logit(prediction[active])
    def objective(parameters):
        candidate = sigmoid(parameters[0] * z + parameters[1])
        return float(np.mean((y - candidate) ** 2))

    result = minimize(
        objective, x0=np.asarray([1., 0.]), method="L-BFGS-B",
        bounds=((.80, 1.20), (-.20, .20)),
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 100},
    )
    if not result.success:
        raise RuntimeError(f"calibration optimization failed: {result.message}")
    return {"scale": float(result.x[0]), "intercept": float(result.x[1])}


def apply_policy(prediction, rows, policy):
    z = logit(prediction)
    output = prediction.copy()
    if policy["kind"] == "global":
        return sigmoid(policy["scale"] * z + policy["intercept"])
    if policy["kind"] == "game_type":
        for game_type in ("R", "F"):
            active = rows["game_type"].eq(game_type).to_numpy()
            values = policy[game_type]
            output[active] = sigmoid(
                values["scale"] * z[active] + values["intercept"]
            )
        return output
    raise ValueError(policy["kind"])


def report(target, base, candidate, rows):
    return {
        name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
        for name, mask in masks(rows).items() if mask.any()
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv",
        usecols=["season", "game_month", "game_type", "control_success"],
        encoding="utf-8-sig", low_memory=False,
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        oof = {key: source[key] for key in source.files}
    folds = {}
    summaries = []
    for year in (2023, 2024):
        active = oof["season"] == year
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        target = oof["target"][active].astype(float)
        prediction = oof["blended"][active].astype(float)
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"v23 rows do not align for {year}")
        folds[year] = (target, prediction, rows)
        summaries.append({
            "year": year, "rows": len(rows), "target_mean": float(target.mean()),
            "prediction_mean": float(prediction.mean()), "bss": bss(target, prediction),
            "regular_target": float(target[rows["game_type"].eq("R")].mean()),
            "regular_prediction": float(prediction[rows["game_type"].eq("R")].mean()),
            "futures_target": float(target[rows["game_type"].eq("F")].mean()),
            "futures_prediction": float(prediction[rows["game_type"].eq("F")].mean()),
        })

    source_target, source_prediction, source_rows = folds[2023]
    valid_target, valid_prediction, valid_rows = folds[2024]
    policies = []
    global_affine = best_affine(source_target, source_prediction)
    policies.append({"kind": "global", **global_affine})
    policies.append({
        "kind": "global", "scale": 1.,
        "intercept": best_affine(source_target, source_prediction)["intercept"],
    })
    game_type_policy = {"kind": "game_type"}
    for game_type in ("R", "F"):
        active = source_rows["game_type"].eq(game_type).to_numpy()
        game_type_policy[game_type] = best_affine(
            source_target, source_prediction, active,
        )
    policies.append(game_type_policy)
    policies.append({
        "kind": "game_type",
        "R": {"scale": 1., "intercept": game_type_policy["R"]["intercept"]},
        "F": {"scale": 1., "intercept": game_type_policy["F"]["intercept"]},
    })

    results = []
    for policy in policies:
        source_candidate = apply_policy(source_prediction, source_rows, policy)
        valid_candidate = apply_policy(valid_prediction, valid_rows, policy)
        gains = {
            "2023_fit": report(
                source_target, source_prediction, source_candidate, source_rows,
            ),
            "2024_transfer": report(
                valid_target, valid_prediction, valid_candidate, valid_rows,
            ),
        }
        results.append({
            "policy": policy, "gains": gains,
            "valid_mean": float(valid_candidate.mean()),
        })
    output = root / "research/v23_forward_calibration.json"
    output.write_text(json.dumps({
        "season_rates": data.groupby(["season", "game_type"], observed=True)[
            "control_success"
        ].agg(["mean", "size"]).reset_index().to_dict("records"),
        "oof_summaries": summaries, "policies": results,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"oof_summaries": summaries, "policies": results}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
