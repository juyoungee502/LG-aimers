"""Promote the conservative train-only calibration over the proven v61 path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_v63_train_trend_calibration import CLIP, OFFSET
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
VERSION = "v63_train_trend_calibration"


def main() -> None:
    report = json.loads(
        (ROOT / "research/v63_train_trend_calibration.json").read_text(encoding="utf-8")
    )
    if not np.isclose(report["configuration"]["probability_offset"], OFFSET):
        raise ValueError("v63 research and deployment offsets differ")
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        v61 = {key: archive[key] for key in archive.files}
    target = v61["target"].astype(float)
    base = v61["blended"].astype(float)
    season = v61["season"].astype(int)
    prediction = base.copy()
    active = season == 2024
    prediction[active] = np.clip(prediction[active] + OFFSET, *CLIP)
    forward_gain = float(
        bss(target[active], prediction[active]) - bss(target[active], base[active])
    )
    if not np.isclose(forward_gain, report["strict_oof_2024_gain"], atol=1e-10):
        raise ValueError("v63 OOF promotion does not match research")
    np.savez_compressed(
        ROOT / "outputs/v63_oof_predictions.npz",
        **{key: value for key, value in v61.items() if key != "blended"},
        blended=prediction,
    )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if "v61_public_complete_shape" not in names:
        raise ValueError("v63 must retain the complete v61 path")
    metadata["model_names"] = [
        name for name in names
        if name not in ("v62_public_residual_frontier", VERSION)
    ] + [VERSION]
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "baseline": "v61_public_complete_shape",
        "probability_offset": OFFSET,
        "offset_space": "probability",
        "applied_after_all_structural_components": True,
        "strict_oof_2024_gain": forward_gain,
        "train_only_rate_forecast": report["configuration"]["train_only_2025_rate_forecast"],
        "proxy_prediction_mean": report["configuration"]["proxy_prediction_mean"],
        "full_train_trend_offset": report["configuration"]["full_train_trend_offset"],
        "fraction_of_full_offset": report["configuration"]["fraction_of_full_offset"],
        "projected_public_score": report["projection"]["score"],
        "projected_public_range": report["projection"]["range"],
        "proxy_uses_training_rows_only": True,
        "leaderboard_inferred_target_rate_used": False,
        "row_independent_inference": True,
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "removed_version": "v62_public_residual_frontier",
        "baseline": "v61_public_complete_shape",
        "probability_offset": OFFSET,
        "strict_oof_2024_gain": forward_gain,
        "projection": report["projection"],
        "leaderboard_inferred_target_rate_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    output = ROOT / "research/v63_promotion.json"
    output.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
