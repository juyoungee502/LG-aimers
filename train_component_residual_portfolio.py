"""Promote the forward-audited component residual portfolio to v22."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_residual_portfolio import apply_component_portfolio
from research_residual_portfolio_v19 import prepare
from residual_portfolio import apply_frozen_portfolio
from train_robust_residual_portfolio import freeze as freeze_v21


SELECTED = (
    ("brier_regressor", "other", .50),
    ("weighted_catboost", "two_strike", .25),
    ("categorical_count_expert", "other", .25),
    ("categorical_count_expert", "regular_count_1", .25),
    ("history_expert", "regular_count_2", .25),
    ("pre_specialist_blend", "regular_runners_1", .25),
    ("pre_specialist_blend", "regular_count_9", .25),
    ("count_expert", "regular_count_1", .25),
    ("count_expert", "regular_count_11", .25),
    ("brier_regressor", "regular_count_1", .25),
    ("categorical_catboost", "regular_count_1", .25),
    ("categorical_count_expert", "regular_runners_2", .25),
)


def bss(y, prediction):
    rate = float(y.mean())
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / (rate * (1. - rate))))


def component_dict(oof, mask):
    names = list(oof["model_names"].astype(str))
    matrix = oof["predictions"][mask].astype(np.float64)
    values = {name: matrix[:, index] for index, name in enumerate(names)}
    values["pre_specialist_blend"] = oof["base_blended"][mask].astype(np.float64)
    return values


def gate(frame, name):
    regular = frame["game_type"].eq("R").to_numpy()
    count = frame["count_state"].to_numpy()
    if name == "other":
        return count % 3 != 2
    if name == "two_strike":
        return count % 3 == 2
    if name.startswith("regular_count_"):
        return regular & (count == int(name.rsplit("_", 1)[1]))
    if name.startswith("regular_runners_"):
        runners = frame["runner_count_code"].to_numpy()
        return regular & (runners == int(name.rsplit("_", 1)[1]))
    raise ValueError(name)


def fit_effects(frame, y, anchor, v21_prediction, components):
    residual = y - v21_prediction
    effects = []
    for prediction_name, gate_name, scale in SELECTED:
        direction = np.nan_to_num(components[prediction_name] - anchor, nan=0.)
        direction *= gate(frame, gate_name)
        finite = np.isfinite(direction)
        denominator = float(np.dot(direction[finite], direction[finite]))
        raw_weight = float(np.clip(
            np.dot(residual[finite], direction[finite]) / max(denominator, 1e-12),
            -1., 1.,
        ))
        effects.append({
            "prediction": prediction_name, "gate": gate_name,
            "weight": raw_weight * scale, "raw_weight": raw_weight,
            "selection_scale": scale,
        })
    return {"effects": effects}


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    with np.load(root / "outputs/v21_oof_predictions.npz") as loaded:
        v21_oof = {key: loaded[key] for key in loaded.files}
    source, valid = oof["season"] == 2023, oof["season"] == 2024
    source_rows = prepare(data.loc[data["season"].eq(2023)].reset_index(drop=True))
    valid_rows = prepare(data.loc[data["season"].eq(2024)].reset_index(drop=True))
    source_y = oof["target"][source].astype(np.float64)
    source_anchor = oof["blended"][source].astype(np.float64)
    source_v21_config = freeze_v21(source_rows, source_y - source_anchor)
    source_v21 = np.clip(
        source_anchor + apply_frozen_portfolio(source_rows, source_v21_config), .005, .995,
    )
    validation_config = fit_effects(
        source_rows, source_y, source_anchor, source_v21, component_dict(oof, source),
    )

    y = oof["target"][valid].astype(np.float64)
    anchor = oof["blended"][valid].astype(np.float64)
    base = v21_oof["blended"][valid].astype(np.float64)
    correction = apply_component_portfolio(
        valid_rows, component_dict(oof, valid), anchor, validation_config,
    )
    prediction = np.clip(base + correction, .005, .995)
    quarter = np.linspace(0, len(y), 5, dtype=int)
    report = {
        "v21_bss": bss(y, base), "v22_bss": bss(y, prediction),
        "gain": bss(y, prediction) - bss(y, base),
        "quarter_gains": [
            bss(y[quarter[i]:quarter[i + 1]], prediction[quarter[i]:quarter[i + 1]])
            - bss(y[quarter[i]:quarter[i + 1]], base[quarter[i]:quarter[i + 1]])
            for i in range(4)
        ],
        "validation_effects": validation_config["effects"],
    }
    if report["v22_bss"] < 984.9 or min(report["gain"], *report["quarter_gains"]) <= 0.:
        raise RuntimeError(f"Component portfolio failed promotion: {report}")
    print(f"v22 validation: {json.dumps(report)}", flush=True)

    deploy_anchor = anchor
    deploy_v21_config = freeze_v21(valid_rows, y - deploy_anchor)
    deploy_v21 = np.clip(
        deploy_anchor + apply_frozen_portfolio(valid_rows, deploy_v21_config), .005, .995,
    )
    deploy_config = fit_effects(
        valid_rows, y, deploy_anchor, deploy_v21, component_dict(oof, valid),
    )
    metadata_path = root / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v21_robust_residual_portfolio":
        raise ValueError(f"Expected v21 metadata, got {metadata.get('version')}")
    metadata["version"] = "v22_component_residual_portfolio"
    metadata["component_residual_portfolio"] = deploy_config
    metadata["training_info"]["v22_validation"] = report
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    upgraded = v21_oof["blended"].astype(np.float64).copy()
    upgraded[valid] = prediction
    output = root / "outputs/v22_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in v21_oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(f"Stored v22 portfolio and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
