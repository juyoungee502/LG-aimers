"""Promote the conservative F-regime v54 scaling policy to v55."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_v40_failure_seed_stability import logit, sigmoid


ROOT = Path(__file__).resolve().parent
R_SCALE = 1.0
F_SCALE = 1.125


def main():
    report_path = ROOT / "research/v55_v54_scaling.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = next(
        row for row in report["ranked"]
        if row["r_scale"] == R_SCALE and row["f_scale"] == F_SCALE
    )
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38_archive = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54_archive = {key: archive[key] for key in archive.files}
    prediction = v54_archive["blended"].astype(np.float64).copy()
    active = v54_archive["season"] == 2024
    v38 = np.clip(v38_archive["blended"][active].astype(float), .005, .995)
    v54 = np.clip(prediction[active], .005, .995)
    import pandas as pd
    train = pd.read_csv(
        ROOT / "data/train.csv", usecols=["season", "game_type"],
        encoding="utf-8-sig",
    )
    futures = train.loc[train["season"].eq(2024), "game_type"].astype(str).eq("F").to_numpy()
    scale = np.where(futures, F_SCALE, R_SCALE)
    prediction[active] = sigmoid(logit(v38) + scale * (logit(v54) - logit(v38)))
    np.savez_compressed(
        ROOT / "outputs/v55_oof_predictions.npz",
        **{key: value for key, value in v54_archive.items() if key != "blended"},
        blended=prediction,
    )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "v54_roster_robust_command" not in metadata.get("model_names", []):
        raise ValueError("v55 requires complete v54 model artifacts")
    retired = {"v60_fraction_confidence", "v67_multitask_tabm"}
    metadata["model_names"] = [
        name for name in metadata["model_names"] if name not in retired
    ]
    if "v55_v54_regime_scaling" not in metadata["model_names"]:
        metadata["model_names"].append("v55_v54_regime_scaling")
    metadata.pop("v60_fraction_confidence", None)
    metadata.pop("v67_multitask_tabm", None)
    metadata["version"] = "v55_v54_regime_scaling"
    metadata["v55_v54_regime_scaling"] = {
        "r_scale": R_SCALE, "f_scale": F_SCALE,
        "anchor": "v54_roster_robust_command",
        "validation": selected,
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    print(json.dumps(metadata["v55_v54_regime_scaling"], indent=2))


if __name__ == "__main__":
    main()
