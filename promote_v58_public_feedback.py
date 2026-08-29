"""Promote the public-feedback v58 counterstep over the v56 anchor."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import logit, sigmoid
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v58_public_feedback_counterstep"
F_SCALE = 1.375
R_V57_MULTIPLIER = -0.25
CLIP = (0.005, 0.995)


def main() -> None:
    report = json.loads(
        (ROOT / "research/v58_public_feedback.json").read_text(encoding="utf-8")
    )
    selected = report["selected"]
    if (
        selected["f_scale"] != F_SCALE
        or selected["r_v57_multiplier"] != R_V57_MULTIPLIER
    ):
        raise ValueError("v58 research and production settings differ")

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v57_oof_predictions.npz") as archive:
        v57 = {key: archive[key] for key in archive.files}
    train = pd.read_csv(
        ROOT / "data/train.csv", usecols=["season", "game_type"],
        encoding="utf-8-sig",
    )
    positions = np.concatenate([
        np.flatnonzero(train["season"].to_numpy() == year)
        for year in (2023, 2024)
    ])
    rows = train.iloc[positions].reset_index(drop=True)
    year = v38["season"].astype(int)
    target = v38["target"].astype(float)
    active = year == 2024
    futures = rows["game_type"].astype(str).eq("F").to_numpy()
    active_f = active & futures

    v56 = v54["blended"].astype(float).copy()
    v56[active_f] = sigmoid(
        logit(v38["blended"][active_f].astype(float))
        + 1.25 * (
            logit(v54["blended"][active_f].astype(float))
            - logit(v38["blended"][active_f].astype(float))
        )
    )
    v57_correction = v57["blended"].astype(float) - v56
    prediction = v54["blended"].astype(float).copy()
    prediction[active_f] = sigmoid(
        logit(v38["blended"][active_f].astype(float))
        + F_SCALE * (
            logit(v54["blended"][active_f].astype(float))
            - logit(v38["blended"][active_f].astype(float))
        )
    )
    prediction += R_V57_MULTIPLIER * v57_correction
    prediction = np.clip(prediction, *CLIP)
    gain = float(
        bss(target[active], prediction[active])
        - bss(target[active], v56[active])
    )
    expected_gain = float(selected["local_gains_over_v56"]["all"])
    if not np.isclose(gain, expected_gain, atol=1e-9):
        raise ValueError(f"v58 OOF reconstruction mismatch: {gain} != {expected_gain}")
    np.savez_compressed(
        ROOT / "outputs/v58_oof_predictions.npz",
        **{key: value for key, value in v54.items() if key != "blended"},
        blended=prediction,
    )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if not {
        "v54_roster_robust_command", "v57_conservative_r_table",
    }.issubset(names):
        raise ValueError("v58 requires the complete v57 staging artifacts")
    v57_configuration = metadata["v57_conservative_r_table"]
    retired = {
        "v55_v54_regime_scaling", "v56_v54_regime_scaling",
        "v57_conservative_r_table", VERSION,
    }
    metadata["model_names"] = [name for name in names if name not in retired]
    metadata["model_names"].append(VERSION)
    for name in retired:
        metadata.pop(name, None)
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "r_scale": 1.0,
        "f_scale": F_SCALE,
        "r_v57_multiplier": R_V57_MULTIPLIER,
        "regular": v57_configuration["regular"],
        "exposure_gate": v57_configuration["exposure_gate"],
        "source_season": 2024,
        "target_season": 2025,
        "anchor": "v54_roster_robust_command",
        "public_anchors": {
            "v55": 1113.6, "v56": 1113.86, "v57_reported": 1112.0,
        },
        "validation": report,
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "version": VERSION,
        "f_scale": F_SCALE,
        "r_v57_multiplier": R_V57_MULTIPLIER,
        "local_gain_over_v56": gain,
        "projected_public_score_range": selected["projected_public_score_range"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
