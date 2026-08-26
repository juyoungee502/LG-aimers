"""Promote the fully quarter-audited frozen residual portfolio to v21."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_residual_portfolio_v19 import prepare
from residual_portfolio import (
    apply_frozen_portfolio,
    freeze_generic_categorical_effect,
    freeze_numeric_effect,
    freeze_pair_effect,
)


NUMERIC = (
    ("derived_recent_success_mean", "none", 8, 3200., .17500),
    ("asof_pitcher_prev3_game_success_rate", "baseout", 16, 50., .07500),
    ("derived_success_trend_1_3", "baseout", 8, 800., .10000),
    ("asof_pitcher_breaking_rate", "count", 8, 200., .01250),
    ("asof_pitcher_prev3_game_middle_rate", "count_hands", 8, 800., -.07500),
    ("asof_pitcher_breaking_rate", "count_hands", 8, 50., .01250),
    ("derived_success_trend_1_3", "count", 16, 3200., -.21875),
    ("derived_recent_middle_mean", "count_hands", 16, 3200., .17500),
    ("asof_pitcher_fastball_rate", "count_hands", 16, 800., .02500),
    ("home_win_expectancy", "baseout", 16, 3200., -.09375),
    ("asof_pitcher_prev1_game_success_rate", "count", 16, 200., .02500),
    ("asof_pitcher_offspeed_rate", "count", 16, 200., .03125),
    ("asof_pitcher_n", "hands", 8, 50., .01250),
    ("asof_pitcher_prev5_game_middle_rate", "count", 16, 3200., -.09375),
    ("asof_pitcher_prev1_game_middle_rate", "count", 8, 200., -.01875),
    ("derived_middle_range", "hands", 16, 200., .01250),
    ("run_total_before", "count", 8, 50., .01875),
)
CATEGORICAL = (
    (("runner_count_code",), 100., .32500),
    (("batter_id", "pitcher_hand"), 400., .06250),
    (("runner_count_code", "count_state"), 200., .06250),
)
PAIRS = (
    (("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate"),
     "count", 8, 50., .02500),
    (("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate"),
     "baseout", 4, 800., .05000),
    (("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate"),
     "baseout", 8, 800., .07500),
    (("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate"),
     "baseout", 4, 50., .01875),
)


def bss(y, prediction):
    rate = float(y.mean())
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / (rate * (1. - rate))))


def freeze(frame, residual):
    effects = [freeze_numeric_effect(frame, residual, *spec) for spec in NUMERIC]
    effects.extend(
        freeze_generic_categorical_effect(frame, residual, *spec)
        for spec in CATEGORICAL
    )
    effects.extend(freeze_pair_effect(frame, residual, *spec) for spec in PAIRS)
    return {"game_type": "R", "effects": effects}


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    source = oof["season"] == 2023
    valid = oof["season"] == 2024
    source_rows = prepare(data.loc[data["season"].eq(2023)].reset_index(drop=True))
    valid_rows = prepare(data.loc[data["season"].eq(2024)].reset_index(drop=True))
    if not np.allclose(source_rows["control_success"], oof["target"][source]):
        raise ValueError("2023 OOF rows differ")
    if not np.allclose(valid_rows["control_success"], oof["target"][valid]):
        raise ValueError("2024 OOF rows differ")

    source_residual = oof["target"][source] - oof["blended"][source]
    validation_config = freeze(source_rows, source_residual)
    correction = apply_frozen_portfolio(valid_rows, validation_config)
    y = oof["target"][valid].astype(np.float64)
    base = oof["blended"][valid].astype(np.float64)
    prediction = np.clip(base + correction, .005, .995)
    quarter = np.linspace(0, len(y), 5, dtype=int)
    report = {
        "base_bss": bss(y, base), "v21_bss": bss(y, prediction),
        "gain": bss(y, prediction) - bss(y, base),
        "quarter_gains": [
            bss(y[quarter[i]:quarter[i + 1]], prediction[quarter[i]:quarter[i + 1]])
            - bss(y[quarter[i]:quarter[i + 1]], base[quarter[i]:quarter[i + 1]])
            for i in range(4)
        ],
    }
    if report["v21_bss"] < 970.9 or min(report["gain"], *report["quarter_gains"]) <= 0.:
        raise RuntimeError(f"Robust portfolio failed promotion: {report}")
    print(f"v21 validation: {json.dumps(report)}", flush=True)

    upgraded = oof["blended"].astype(np.float64).copy()
    upgraded[valid] = prediction
    deploy_rows = prepare(data.loc[data["season"].eq(2024)].reset_index(drop=True))
    deploy_residual = oof["target"][valid] - oof["blended"][valid]
    deploy_config = freeze(deploy_rows, deploy_residual)
    metadata_path = root / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v20_residual_portfolio":
        raise ValueError(f"Expected v20 metadata, got {metadata.get('version')}")
    metadata["version"] = "v21_robust_residual_portfolio"
    metadata["residual_portfolio"] = deploy_config
    metadata["training_info"]["v21_validation"] = report
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    output = root / "outputs/v21_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(f"Stored v21 portfolio and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
