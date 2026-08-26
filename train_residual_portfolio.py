"""Promote the robust frozen residual portfolio to v20."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from residual_portfolio import (
    apply_frozen_portfolio,
    freeze_categorical_effect,
    freeze_numeric_effect,
)


NUMERIC = (
    ("asof_pitcher_prev3_game_success_rate", "baseout", 16, 50., .10000),
    ("asof_pitcher_prev1_game_success_rate", "none", 16, 3200., .27500),
    ("asof_pitcher_prev1_game_success_rate", "baseout", 16, 50., .05000),
    ("asof_pitcher_pitchmix_n", "baseout", 8, 3200., .15000),
    ("run_total_before", "count", 8, 50., .07500),
    ("asof_pitcher_breaking_rate", "count_hands", 8, 50., .03750),
    ("asof_pitcher_prev5_game_middle_rate", "count", 16, 3200., -.09375),
    ("asof_pitcher_prev3_game_middle_rate", "count_hands", 8, 800., -.09375),
    ("asof_pitcher_prev1_game_middle_rate", "count", 8, 200., -.09375),
)


def bss(y, prediction):
    rate = float(y.mean())
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / (rate * (1. - rate))))


def freeze(frame, residual):
    effects = [freeze_numeric_effect(frame, residual, *spec) for spec in NUMERIC]
    # Greedy audit selected a 1.25 multiplier on the original .125 effect.
    effects.insert(1, freeze_categorical_effect(frame, residual, 400., .15625))
    return {"game_type": "R", "effects": effects}


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    upgraded = oof["blended"].astype(np.float64).copy()
    source = oof["season"] == 2023
    valid = oof["season"] == 2024
    source_rows = data.loc[data["season"].eq(2023)].reset_index(drop=True)
    valid_rows = data.loc[data["season"].eq(2024)].reset_index(drop=True)
    if not np.allclose(source_rows["control_success"], oof["target"][source]):
        raise ValueError("2023 OOF rows differ")
    if not np.allclose(valid_rows["control_success"], oof["target"][valid]):
        raise ValueError("2024 OOF rows differ")
    source_residual = oof["target"][source] - oof["blended"][source]
    validation_config = freeze(source_rows, source_residual)
    correction = apply_frozen_portfolio(valid_rows, validation_config)
    base = upgraded[valid].copy()
    prediction = np.clip(base + correction, .005, .995)
    y = oof["target"][valid].astype(np.float64)
    quarter = np.linspace(0, len(y), 5, dtype=int)
    report = {
        "base_bss": bss(y, base), "v20_bss": bss(y, prediction),
        "gain": bss(y, prediction) - bss(y, base),
        "quarter_gains": [
            bss(y[quarter[i]:quarter[i + 1]], prediction[quarter[i]:quarter[i + 1]])
            - bss(y[quarter[i]:quarter[i + 1]], base[quarter[i]:quarter[i + 1]])
            for i in range(4)
        ],
    }
    if min(report["gain"], *report["quarter_gains"]) <= 0.:
        raise RuntimeError(f"Residual portfolio failed promotion: {report}")
    upgraded[valid] = prediction
    print(f"v20 validation: {json.dumps(report)}", flush=True)

    deploy_rows = data.loc[data["season"].eq(2024)].reset_index(drop=True)
    deploy_residual = oof["target"][valid] - oof["blended"][valid]
    deploy_config = freeze(deploy_rows, deploy_residual)
    metadata_path = root / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v19_failure_specialist":
        raise ValueError(f"Expected v19 metadata, got {metadata.get('version')}")
    metadata["version"] = "v20_residual_portfolio"
    metadata["residual_portfolio"] = deploy_config
    metadata["training_info"]["v20_validation"] = report
    names = metadata.setdefault("model_names", [])
    if "residual_portfolio" not in names:
        names.append("residual_portfolio")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    output = root / "outputs/v20_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(f"Stored v20 portfolio and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
