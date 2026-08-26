"""Fit a nonlinear previous-season correction to v19 OOF residuals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def params(seed, depth):
    return dict(
        iterations=900, learning_rate=.02, depth=depth, loss_function="RMSE",
        eval_metric="RMSE", l2_leaf_reg=300., random_strength=1.5,
        border_count=32, task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    seasons = data["season"].to_numpy(np.int16)
    source = oof["season"] == 2023
    valid_oof = oof["season"] == 2024
    train = seasons == 2023
    valid = seasons == 2024
    if not np.allclose(target[train], oof["target"][source]):
        raise ValueError("2023 residual rows do not align")
    if not np.allclose(target[valid], oof["target"][valid_oof]):
        raise ValueError("2024 rows do not align")
    residual = oof["target"][source] - oof["blended"][source]
    predictions, names = [], []
    for depth in (4, 6):
        for gate_name in ("all", "R"):
            fit = train.copy()
            if gate_name == "R":
                fit &= data["game_type"].eq("R").to_numpy()
            model = CatBoostRegressor(**params(1700 + depth, depth))
            model.fit(features.loc[fit], residual[data.loc[train, "game_type"].eq("R").to_numpy()] if gate_name == "R" else residual)
            predictions.append(model.predict(features.loc[valid]))
            names.append(f"residual_d{depth}_{gate_name}")
            print(f"Residual booster complete: {names[-1]}, rows={fit.sum()}", flush=True)

    y = oof["target"][valid_oof].astype(float)
    base = oof["blended"][valid_oof].astype(float)
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2
    reports = []
    for name, correction in zip(names, predictions):
        for apply_gate in ("all", "R"):
            active = np.ones(len(y), bool) if apply_gate == "all" else regular
            for weight in np.arange(-.25, 1.001, .025):
                prediction = base.copy()
                prediction[active] = np.clip(
                    prediction[active] + weight * correction[active], .005, .995,
                )
                report = {
                    "name": name, "apply_gate": apply_gate, "weight": float(weight),
                    "gain": bss(y, prediction) - bss(y, base),
                    "gain_first_half": bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                    "gain_second_half": bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                    "correction_mean": float(correction[active].mean()),
                    "correction_std": float(correction[active].std()),
                }
                report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
                reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / "research/residual_booster_2024.npz"
    np.savez_compressed(
        output, names=np.asarray(names), predictions=np.column_stack(predictions).astype(np.float32),
        target=y.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps(reports[:60], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
