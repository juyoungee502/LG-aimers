"""Forward-audit an F-regime model enriched by honest base-model predictions.

The source is 2023 F and the untouched validation season is 2024 F.  All stack
features are rolling OOF predictions produced without the row's target.  The
script is diagnostic only and never mutates submission artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
SEEDS = (1812, 2025, 3407)
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup",
    "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(args, seed):
    values = dict(
        iterations=1000, learning_rate=.025, depth=6,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=100.,
        random_strength=1., random_seed=seed, border_count=32,
        allow_writing_files=False, verbose=0, task_type=args.task_type,
        thread_count=args.threads,
    )
    if args.task_type == "GPU":
        values["devices"] = args.devices
    return values


def stack_features(oof: dict[str, np.ndarray]) -> pd.DataFrame:
    names = oof["model_names"].astype(str).tolist()
    matrix = np.clip(oof["predictions"].astype(float), 1e-5, 1. - 1e-5)
    output = pd.DataFrame(index=np.arange(len(matrix)))
    for index, name in enumerate(names):
        output[f"stack_{name}"] = matrix[:, index]
        output[f"stack_logit_{name}"] = logit(matrix[:, index])
    output["stack_component_mean"] = matrix.mean(axis=1)
    output["stack_component_std"] = matrix.std(axis=1)
    output["stack_component_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    for name in ("base_blended", "trackman_context", "blended"):
        values = np.clip(oof[name].astype(float), 1e-5, 1. - 1e-5)
        output[f"stack_{name}"] = values
        output[f"stack_logit_{name}"] = logit(values)
    output["stack_base_minus_mean"] = (
        output["stack_base_blended"] - output["stack_component_mean"]
    )
    output["stack_trackman_minus_base"] = (
        output["stack_trackman_context"] - output["stack_base_blended"]
    )
    output["stack_v17_minus_base"] = (
        output["stack_blended"] - output["stack_base_blended"]
    )
    return output.astype(np.float32)


def segment_masks(rows: pd.DataFrame):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        **{
            f"month_{month}": rows["game_month"].eq(month).to_numpy()
            for month in sorted(rows["game_month"].unique())
        },
    }


def gains(target, base, candidate, rows):
    return {
        name: bss(target[active], candidate[active]) - bss(target[active], base[active])
        for name, active in segment_masks(rows).items() if active.any()
    }


def main():
    args = arguments()
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    core = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(core, raw)
    core = add_state_interactions(core)
    for column in CAT_COLUMNS:
        core[column] = core[column].fillna(-1).astype(np.int32)

    with np.load(ROOT / "outputs/v17_oof_predictions.npz") as archive:
        v17 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        v24 = {key: archive[key] for key in archive.files}
    stack = stack_features(v17)
    oof_rows = pd.concat(
        [raw.loc[raw["season"].eq(year)] for year in (2023, 2024)],
        ignore_index=True,
    )
    if len(oof_rows) != len(stack) or not np.allclose(
        oof_rows.index.to_numpy(), stack.index.to_numpy(),
    ):
        raise ValueError("OOF stack row count differs")
    if not np.allclose(v17["target"], v24["target"]):
        raise ValueError("v17 and v24 targets differ")
    oof_positions = np.concatenate([
        np.flatnonzero(raw["season"].eq(year).to_numpy()) for year in (2023, 2024)
    ])
    if not np.allclose(target[oof_positions], v17["target"]):
        raise ValueError("OOF rows and targets differ")

    # Map the compact OOF stack back to the corresponding full-frame positions.
    stack.index = oof_positions
    enriched = pd.concat([core, stack.reindex(core.index)], axis=1)
    seasons = raw["season"].to_numpy(np.int16)
    futures = raw["game_type"].eq("F").to_numpy()
    train = (seasons == 2023) & futures
    valid = (seasons == 2024) & futures
    reference_members = []
    enriched_members = []
    for seed in SEEDS:
        reference = CatBoostClassifier(**parameters(args, seed))
        reference.fit(core.loc[train], target[train])
        reference_members.append(reference.predict_proba(core.loc[valid])[:, 1])
        print(f"F stack reference complete: seed={seed}", flush=True)
        member = CatBoostClassifier(**parameters(args, seed))
        member.fit(enriched.loc[train], target[train])
        enriched_members.append(member.predict_proba(enriched.loc[valid])[:, 1])
        print(f"F stack enriched complete: seed={seed}", flush=True)
    reference = np.mean(reference_members, axis=0)
    candidate = np.mean(enriched_members, axis=0)

    v24_fold = v24["season"] == 2024
    v24_rows = raw.loc[seasons == 2024].reset_index(drop=True)
    v24_f = v24_rows["game_type"].eq("F").to_numpy()
    y = v24["target"][v24_fold][v24_f].astype(float)
    base = v24["blended"][v24_fold][v24_f].astype(float)
    rows = v24_rows.loc[v24_f].reset_index(drop=True)
    if not np.allclose(y, target[valid]):
        raise ValueError("F validation rows differ")

    directions = {
        "paired_delta": logit(candidate) - logit(reference),
        "direct_enriched": logit(candidate) - logit(base),
    }
    reports = []
    for name, direction in directions.items():
        for weight in np.arange(-.50, 1.501, .025):
            prediction = sigmoid(logit(base) + weight * direction)
            report = {
                "name": name, "weight": float(weight),
                "gains": gains(y, base, prediction, rows),
            }
            report["min_half"] = min(
                report["gains"]["half_1"], report["gains"]["half_2"],
            )
            report["min_quarter"] = min(
                report["gains"][f"q{index}"] for index in range(1, 5)
            )
            report["min_month"] = min(
                value for key, value in report["gains"].items()
                if key.startswith("month_")
            )
            reports.append(report)
    reports.sort(
        key=lambda value: (
            min(value["min_half"], value["min_quarter"], value["min_month"]),
            value["gains"]["all"],
        ), reverse=True,
    )
    output = ROOT / "research/v24_f_stack_2024.npz"
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        reference=reference.astype(np.float32), candidate=candidate.astype(np.float32),
        paired_delta=directions["paired_delta"].astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "reference_bss": bss(y, reference), "enriched_bss": bss(y, candidate),
        "paired_delta_std": float(directions["paired_delta"].std()),
        "top": reports[:40],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
