"""Promote the forward-audited probability-shape portfolio to v23."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_residual_portfolio import apply_component_portfolio
from probability_residual_portfolio import apply_probability_portfolio
from research_residual_portfolio_v19 import prepare
from residual_portfolio import apply_frozen_portfolio
from train_component_residual_portfolio import component_dict, fit_effects
from train_robust_residual_portfolio import freeze as freeze_v21


SELECTED = (
    ("uncertainty", "regular_count_2", .75),
    ("quadratic", "regular_count_3", .25),
    ("constant", "regular_count_9", .25),
    ("uncertainty", "regular_count_3", .25),
    ("uncertainty", "regular_count_5", .25),
    ("quadratic", "regular_runners_2", .25),
    ("quadratic", "regular_count_5", .25),
    ("uncertainty", "regular_runners_2", .25),
    ("uncertainty", "regular_count_4", .25),
    ("quadratic", "regular_count_4", .25),
    ("quadratic", "regular_count_2", .75),
    ("uncertainty", "regular_count_11", .25),
    ("linear", "regular_count_9", .25),
)


def bss(y, prediction):
    rate = float(y.mean())
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / (rate * (1. - rate))))


def gate(frame, name):
    regular = frame["game_type"].eq("R").to_numpy()
    count = frame["count_state"].to_numpy()
    if name.startswith("regular_count_"):
        return regular & (count == int(name.rsplit("_", 1)[1]))
    if name.startswith("regular_runners_"):
        return regular & (
            frame["runner_count_code"].to_numpy() == int(name.rsplit("_", 1)[1])
        )
    raise ValueError(name)


def shape(prediction, name):
    p = np.clip(np.asarray(prediction, dtype=np.float64), .005, .995)
    if name == "constant":
        return np.ones(len(p), dtype=np.float64)
    if name == "uncertainty":
        return p * (1. - p)
    if name == "quadratic":
        return (p - .5) ** 2
    if name == "linear":
        return p - .5
    raise ValueError(name)


def fit_probability_effects(frame, y, prediction):
    residual = y - prediction
    effects = []
    for shape_name, gate_name, scale in SELECTED:
        active = gate(frame, gate_name)
        raw_value = shape(prediction, shape_name)
        center = 0. if shape_name == "constant" else float(raw_value[active].mean())
        value = (raw_value - center) * active
        denominator = float(np.dot(value, value))
        raw_weight = float(np.clip(
            np.dot(residual, value) / max(denominator, 1e-12), -1., 1.,
        ))
        effects.append({
            "shape": shape_name, "gate": gate_name, "center": center,
            "weight": raw_weight * scale, "raw_weight": raw_weight,
            "selection_scale": scale,
        })
    return {"effects": effects}


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    with np.load(root / "outputs/v22_oof_predictions.npz") as loaded:
        v22_oof = {key: loaded[key] for key in loaded.files}
    source, valid = oof["season"] == 2023, oof["season"] == 2024
    source_rows = prepare(data.loc[data["season"].eq(2023)].reset_index(drop=True))
    valid_rows = prepare(data.loc[data["season"].eq(2024)].reset_index(drop=True))
    source_y = oof["target"][source].astype(np.float64)
    source_anchor = oof["blended"][source].astype(np.float64)
    source_components = component_dict(oof, source)
    source_v21_config = freeze_v21(source_rows, source_y - source_anchor)
    source_v21 = np.clip(
        source_anchor + apply_frozen_portfolio(source_rows, source_v21_config), .005, .995,
    )
    source_component_config = fit_effects(
        source_rows, source_y, source_anchor, source_v21, source_components,
    )
    source_v22 = np.clip(
        source_v21 + apply_component_portfolio(
            source_rows, source_components, source_anchor, source_component_config,
        ), .005, .995,
    )
    validation_config = fit_probability_effects(source_rows, source_y, source_v22)

    y = oof["target"][valid].astype(np.float64)
    base = v22_oof["blended"][valid].astype(np.float64)
    correction = apply_probability_portfolio(valid_rows, base, validation_config)
    prediction = np.clip(base + correction, .005, .995)
    quarter = np.linspace(0, len(y), 5, dtype=int)
    report = {
        "v22_bss": bss(y, base), "v23_bss": bss(y, prediction),
        "gain": bss(y, prediction) - bss(y, base),
        "quarter_gains": [
            bss(y[quarter[i]:quarter[i + 1]], prediction[quarter[i]:quarter[i + 1]])
            - bss(y[quarter[i]:quarter[i + 1]], base[quarter[i]:quarter[i + 1]])
            for i in range(4)
        ],
        "validation_effects": validation_config["effects"],
    }
    if report["v23_bss"] < 989.5 or min(report["gain"], *report["quarter_gains"]) <= 0.:
        raise RuntimeError(f"Probability portfolio failed promotion: {report}")
    print(f"v23 validation: {json.dumps(report)}", flush=True)

    metadata_path = root / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") not in (
        "v22_component_residual_portfolio", "v23_probability_residual_portfolio",
    ):
        raise ValueError(f"Expected v22/v23 metadata, got {metadata.get('version')}")
    anchor = oof["blended"][valid].astype(np.float64)
    components = component_dict(oof, valid)
    deploy_v21 = np.clip(
        anchor + apply_frozen_portfolio(valid_rows, metadata["residual_portfolio"]),
        .005, .995,
    )
    deploy_v22 = np.clip(
        deploy_v21 + apply_component_portfolio(
            valid_rows, components, anchor, metadata["component_residual_portfolio"],
        ), .005, .995,
    )
    deploy_config = fit_probability_effects(valid_rows, y, deploy_v22)
    metadata["version"] = "v23_probability_residual_portfolio"
    metadata["probability_residual_portfolio"] = deploy_config
    metadata["training_info"]["v23_validation"] = report
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    upgraded = v22_oof["blended"].astype(np.float64).copy()
    upgraded[valid] = prediction
    output = root / "outputs/v23_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in v22_oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(f"Stored v23 portfolio and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
