"""Upgrade trained v11 artifacts to v12 using transferred OOF residual effects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from residual_effects import apply_residual_effects, build_residual_effects


TARGET = "control_success"
SOURCE_COLUMNS = [
    "season", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "balls_before", "strikes_before", "num_runners_on", TARGET,
]


def score(y, prediction):
    rate = float(np.mean(y))
    return 100000. * (1. - np.mean((y - np.clip(prediction, .005, .995)) ** 2) / (rate * (1. - rate)))


def main():
    root = Path(__file__).resolve().parent
    metadata_path = root / "submit" / "model" / "metadata.json"
    diagnostic_path = root / "outputs" / "v11_oof_predictions.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v11_brier_regression":
        raise ValueError(f"Expected v11 artifacts, found {metadata.get('version')}")
    data = np.load(diagnostic_path, allow_pickle=False)
    train = pd.read_csv(root / "data" / "train.csv", usecols=SOURCE_COLUMNS, encoding="utf-8-sig")
    rows = pd.concat([train.loc[train.season.eq(year)] for year in (2023, 2024)], ignore_index=True)
    target = data["target"].astype(np.float64)
    base = data["blended"].astype(np.float64)
    years = data["season"]
    if len(rows) != len(target) or not np.allclose(rows[TARGET].to_numpy(), target):
        raise ValueError("OOF rows do not align with train.csv")

    source = years == 2023
    latest = years == 2024
    validation_effects = build_residual_effects(rows.loc[source].reset_index(drop=True), target[source] - base[source])
    validation_adjustment, _ = apply_residual_effects(rows.loc[latest].reset_index(drop=True), validation_effects)
    corrected = base.copy()
    corrected[latest] = np.clip(base[latest] + validation_adjustment, .005, .995)
    base_score = score(target[latest], base[latest])
    corrected_score = score(target[latest], corrected[latest])
    print(f"v11 2024 BSS={base_score:.4f}; v12 transferred residual BSS={corrected_score:.4f}; delta={corrected_score-base_score:+.4f}")

    final_effects = build_residual_effects(rows, target - base)
    metadata["version"] = "v12_transferred_residual_effects"
    metadata["residual_effects"] = final_effects
    metadata["training_info"]["v12_validation"] = {
        "source_year": 2023, "target_year": 2024,
        "base_bss": base_score, "corrected_bss": corrected_score,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        root / "outputs" / "v12_oof_predictions.npz",
        predictions=data["predictions"], target=target.astype(np.float32),
        season=years, model_names=data["model_names"], two_strike=data["two_strike"],
        base_blended=base, blended=corrected,
    )
    print(f"Upgraded artifacts: {metadata_path}")


if __name__ == "__main__":
    main()
