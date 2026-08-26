"""Validate Trackman learning-using-privileged-information distillation.

Current-pitch Trackman measurements are visible only to a teacher during
training.  The submitted student receives the same pre-pitch, row-local inputs
as the ordinary model.  Teacher labels for student training are two-fold OOF,
so the teacher never predicts a row whose target it fitted.
"""
from __future__ import annotations

import argparse
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


PHYSICAL_COLUMNS = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
PITCH_GROUPS = ("breaking", "fastball", "offspeed", "other")
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed: int, teacher: bool):
    return dict(
        iterations=1000 if teacher else 1400,
        learning_rate=.02 if teacher else .01631820635235777,
        depth=8, l2_leaf_reg=300. if teacher else 509.6419153575998,
        random_strength=2., bagging_temperature=.36881602504480515,
        border_count=32, bootstrap_type="Bayesian", loss_function="RMSE",
        eval_metric="RMSE", task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def aligned_privileged_features(root: Path, rows: pd.DataFrame):
    """Return current-pitch Trackman values aligned to main training rows."""
    with np.load(root / "outputs/trackman_pitch_alignment.npz") as loaded:
        links = pd.DataFrame({
            "row_id": loaded["row_id"].astype(str),
            "trackman_id": loaded["trackman_id"].astype(str),
        })
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=["trackman_id", "pitch_type_group", *PHYSICAL_COLUMNS],
        encoding="utf-8-sig", low_memory=False,
    )
    trackman["trackman_id"] = trackman["trackman_id"].astype(str)
    aligned = links.merge(trackman, on="trackman_id", how="left", validate="one_to_one")
    row_positions = pd.Series(
        np.arange(len(rows), dtype=np.int64), index=rows["row_id"].astype(str),
    )
    aligned["_position"] = aligned["row_id"].map(row_positions)
    aligned = aligned.dropna(subset=["_position"]).copy()
    positions = aligned.pop("_position").astype(np.int64).to_numpy()
    privileged = pd.DataFrame(
        np.nan, index=rows.index,
        columns=[
            *(f"lupi_{column}" for column in PHYSICAL_COLUMNS),
            *(f"lupi_pitch_{group}" for group in PITCH_GROUPS),
        ], dtype=np.float32,
    )
    privileged.loc[positions, [f"lupi_{column}" for column in PHYSICAL_COLUMNS]] = (
        aligned[list(PHYSICAL_COLUMNS)].to_numpy(np.float32)
    )
    for group in PITCH_GROUPS:
        privileged.loc[positions, f"lupi_pitch_{group}"] = (
            aligned["pitch_type_group"].eq(group).to_numpy(np.float32)
        )
    aligned_mask = np.zeros(len(rows), dtype=bool)
    aligned_mask[positions] = True
    return privileged, aligned_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--teacher-weight", type=float, default=.5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    row_hash = pd.util.hash_pandas_object(
        data["row_id"].astype(str), index=False,
    ).to_numpy(np.uint64)
    privileged, aligned = aligned_privileged_features(root, data)

    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    teacher_features = pd.concat([features, privileged], axis=1)
    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    fold_id = (row_hash % 2).astype(np.int8)

    teacher_oof = np.full(len(data), np.nan, dtype=np.float32)
    privileged_valid_predictions = []
    for fold in (0, 1):
        teacher_train = train & aligned & (fold_id != fold)
        teacher_holdout = train & aligned & (fold_id == fold)
        teacher = CatBoostRegressor(**parameters(3100 + fold, teacher=True))
        teacher.fit(teacher_features.loc[teacher_train], target[teacher_train])
        teacher_oof[teacher_holdout] = np.clip(
            teacher.predict(teacher_features.loc[teacher_holdout]), .005, .995,
        )
        privileged_valid_predictions.append(np.clip(
            teacher.predict(teacher_features.loc[valid & aligned]), .005, .995,
        ))
        print(
            f"Teacher fold complete: year={args.valid_year}, fold={fold}, "
            f"train={teacher_train.sum()}, holdout={teacher_holdout.sum()}",
            flush=True,
        )
    if not np.isfinite(teacher_oof[train & aligned]).all():
        raise RuntimeError("Teacher OOF coverage is incomplete")

    soft_target = target.copy()
    distill_rows = train & aligned
    soft_target[distill_rows] = (
        (1. - args.teacher_weight) * target[distill_rows]
        + args.teacher_weight * teacher_oof[distill_rows]
    )
    seed = 3200 + args.valid_year
    reference_model = CatBoostRegressor(**parameters(seed, teacher=False))
    reference_model.fit(features.loc[train], target[train])
    reference = np.clip(reference_model.predict(features.loc[valid]), .005, .995)
    print(f"Reference student complete: {args.valid_year}", flush=True)
    distilled_model = CatBoostRegressor(**parameters(seed, teacher=False))
    distilled_model.fit(features.loc[train], soft_target[train])
    distilled = np.clip(distilled_model.predict(features.loc[valid]), .005, .995)
    print(f"Distilled student complete: {args.valid_year}", flush=True)

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    oof_fold = oof["season"] == args.valid_year
    y = oof["target"][oof_fold].astype(np.float64)
    base = oof["blended"][oof_fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    delta = logit(distilled) - logit(reference)
    midpoint = len(y) // 2
    reports = []
    for mode in ("paired_delta", "direct"):
        for weight in np.arange(-.25 if mode == "paired_delta" else 0.,
                                1.001 if mode == "paired_delta" else .301, .025):
            prediction = base.copy()
            if mode == "paired_delta":
                prediction[regular] = sigmoid(
                    logit(base[regular]) + weight * delta[regular]
                )
            else:
                prediction[regular] = sigmoid(
                    (1. - weight) * logit(base[regular])
                    + weight * logit(distilled[regular])
                )
            report = {
                "mode": mode, "weight": float(weight),
                "gain": bss(y, prediction) - bss(y, base),
                "gain_first_half": bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                "gain_second_half": bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                "reference_bss_R": bss(y[regular], reference[regular]),
                "distilled_bss_R": bss(y[regular], distilled[regular]),
                "delta_std_R": float(delta[regular].std()),
            }
            report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
            reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)

    teacher_valid_mask = valid & aligned
    teacher_valid = np.mean(privileged_valid_predictions, axis=0)
    teacher_y = target[teacher_valid_mask]
    output = root / f"research/lupi_distillation_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        reference=reference.astype(np.float32), distilled=distilled.astype(np.float32),
        delta=delta.astype(np.float32), reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "year": args.valid_year, "teacher_weight": args.teacher_weight,
        "aligned_train_rows": int((train & aligned).sum()),
        "aligned_valid_rows": int(teacher_valid_mask.sum()),
        "teacher_oof_train_bss": float(bss(
            target[train & aligned], teacher_oof[train & aligned],
        )),
        "privileged_teacher_valid_bss": float(bss(teacher_y, teacher_valid)),
        "top": reports[:30],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
