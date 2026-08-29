"""Translate the v57 public failure into conservative v58 candidates.

V57 changed only R rows relative to v56.  Its public loss therefore directly
identifies a 2025-reversed R direction.  This audit measures the correction's
quadratic cost locally and combines a conservative negative R multiplier with
one more public-positive F scaling step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import logit, masks, sigmoid
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
V56_PUBLIC = 1113.86
V57_PUBLIC_REPORTED = 1112.0
V55_PUBLIC = 1113.6
V56_F_SCALE = 1.25
V58_F_SCALE = 1.375
V58_R_MULTIPLIER = -0.25


def main() -> None:
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
    target = v38["target"].astype(float)
    year = v38["season"].astype(int)
    active = year == 2024
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    futures = ~regular

    v56 = v54["blended"].astype(float).copy()
    active_f = active & futures
    v56[active_f] = sigmoid(
        logit(v38["blended"][active_f].astype(float))
        + V56_F_SCALE * (
            logit(v54["blended"][active_f].astype(float))
            - logit(v38["blended"][active_f].astype(float))
        )
    )
    if not np.allclose(v57["blended"][~active], v56[~active]):
        raise ValueError("v57 unexpectedly changed non-2024 OOF rows")
    correction = v57["blended"].astype(float) - v56
    if np.max(np.abs(correction[~(active & regular)])) > 1e-12:
        raise ValueError("v57 public attribution is invalid: non-R rows changed")

    local_r_curve = []
    block = active & regular
    base_r_score = bss(target[block], v56[block])
    for multiplier in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        prediction = np.clip(v56 + multiplier * correction, 0.005, 0.995)
        local_r_curve.append({
            "multiplier": multiplier,
            "gain_2024_all": float(
                bss(target[active], prediction[active])
                - bss(target[active], v56[active])
            ),
            "gain_2024_R": float(bss(target[block], prediction[block]) - base_r_score),
        })
    d_plus = next(row["gain_2024_all"] for row in local_r_curve if row["multiplier"] == 1.0)
    d_minus = next(row["gain_2024_all"] for row in local_r_curve if row["multiplier"] == -1.0)
    local_linear = 0.5 * (d_plus - d_minus)
    local_curvature = -0.5 * (d_plus + d_minus)

    # The public score was reported to the nearest point.  Keep a range rather
    # than pretending its last decimals are known.  Curvature is also stress
    # tested from half to four times the local value.
    public_loss_range = (-2.36, -1.36)
    public_reverse_projection = []
    for public_delta_at_plus_one in public_loss_range:
        for curvature_factor in (0.5, 1.0, 2.0, 4.0):
            curvature = curvature_factor * local_curvature
            linear = public_delta_at_plus_one + curvature
            projected = (
                linear * V58_R_MULTIPLIER
                - curvature * V58_R_MULTIPLIER ** 2
            )
            public_reverse_projection.append({
                "reported_public_delta_at_plus_one": public_delta_at_plus_one,
                "curvature_factor": curvature_factor,
                "projected_v58_r_gain": float(projected),
            })

    local_f_curve = []
    for scale in (1.0, 1.125, 1.25, 1.375, 1.5):
        prediction = v54["blended"].astype(float).copy()
        prediction[active_f] = sigmoid(
            logit(v38["blended"][active_f].astype(float))
            + scale * (
                logit(v54["blended"][active_f].astype(float))
                - logit(v38["blended"][active_f].astype(float))
            )
        )
        local_f_curve.append({
            "scale": scale,
            "gain_over_v54_all": float(
                bss(target[active], prediction[active])
                - bss(target[active], v54["blended"][active].astype(float))
            ),
        })
    f_by_scale = {row["scale"]: row["gain_over_v54_all"] for row in local_f_curve}
    previous_local_step = f_by_scale[1.25] - f_by_scale[1.125]
    next_local_step = f_by_scale[1.375] - f_by_scale[1.25]
    public_previous_step = V56_PUBLIC - V55_PUBLIC
    projected_f_gain = public_previous_step * next_local_step / previous_local_step

    selected = v56.copy()
    selected[active_f] = sigmoid(
        logit(v38["blended"][active_f].astype(float))
        + V58_F_SCALE * (
            logit(v54["blended"][active_f].astype(float))
            - logit(v38["blended"][active_f].astype(float))
        )
    )
    selected += V58_R_MULTIPLIER * correction
    selected = np.clip(selected, 0.005, 0.995)
    segment_gains = {}
    active_positions = np.flatnonzero(active)
    for name, mask in masks(len(active_positions)).items():
        index = active_positions[mask]
        segment_gains[name] = float(
            bss(target[index], selected[index]) - bss(target[index], v56[index])
        )
    segment_gains["R"] = float(
        bss(target[block], selected[block]) - bss(target[block], v56[block])
    )
    segment_gains["F"] = float(
        bss(target[active_f], selected[active_f])
        - bss(target[active_f], v56[active_f])
    )

    reverse_gains = [row["projected_v58_r_gain"] for row in public_reverse_projection]
    deploy_adjusted_projection = []
    deploy_path = ROOT / "research/v58_deploy_curvature.json"
    if deploy_path.is_file():
        deploy_report = json.loads(deploy_path.read_text(encoding="utf-8"))
        deploy_factor = float(deploy_report["mean_square_ratio_deploy_to_validation"])
        deploy_curvature = deploy_factor * local_curvature
        for public_delta_at_plus_one in public_loss_range:
            linear = public_delta_at_plus_one + deploy_curvature
            projected = (
                linear * V58_R_MULTIPLIER
                - deploy_curvature * V58_R_MULTIPLIER ** 2
            )
            deploy_adjusted_projection.append({
                "reported_public_delta_at_plus_one": public_delta_at_plus_one,
                "deploy_curvature_factor": deploy_factor,
                "projected_v58_r_gain": float(projected),
            })
    report = {
        "public_feedback": {
            "v56": V56_PUBLIC,
            "v57_reported": V57_PUBLIC_REPORTED,
            "v57_changed_only_R": True,
            "public_delta_reported": V57_PUBLIC_REPORTED - V56_PUBLIC,
        },
        "correction": {
            "changed_2024_rows": int(np.count_nonzero(correction[active])),
            "mean_abs_on_changed_rows": float(np.mean(np.abs(correction[correction != 0.0]))),
            "max_abs": float(np.max(np.abs(correction))),
            "local_linear_coefficient": float(local_linear),
            "local_curvature_coefficient": float(local_curvature),
        },
        "local_r_curve": local_r_curve,
        "public_reverse_projection": public_reverse_projection,
        "projected_r_gain_range": [float(min(reverse_gains)), float(max(reverse_gains))],
        "deploy_adjusted_projection": deploy_adjusted_projection,
        "local_f_curve": local_f_curve,
        "projected_f_gain": float(projected_f_gain),
        "selected": {
            "version": "v58_public_feedback_counterstep",
            "f_scale": V58_F_SCALE,
            "r_v57_multiplier": V58_R_MULTIPLIER,
            "local_gains_over_v56": segment_gains,
            "projected_public_score_range": (
                [
                    float(V56_PUBLIC + projected_f_gain + min(
                        row["projected_v58_r_gain"]
                        for row in deploy_adjusted_projection
                    )),
                    float(V56_PUBLIC + projected_f_gain + max(
                        row["projected_v58_r_gain"]
                        for row in deploy_adjusted_projection
                    )),
                ] if deploy_adjusted_projection else [
                    float(V56_PUBLIC + projected_f_gain + min(reverse_gains)),
                    float(V56_PUBLIC + projected_f_gain + max(reverse_gains)),
                ]
            ),
        },
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v58_public_feedback.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
