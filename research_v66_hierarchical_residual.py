"""Evaluate a clean-room hierarchical residual ensemble against the v64 anchor."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_v25_temporal_portfolio import bss
from v66_hierarchical_residual import (
    BASE_CATEGORICAL, CLIP, TARGET_COL, build_features, build_snapshots,
)


ROOT = Path(__file__).resolve().parent
SEEDS = (6600, 6617, 6643)
MODEL_SPECS = (
    {"name": "compact_recent", "extended": False, "decay": 0.55,
     "iterations": 140, "seed_offset": 0},
    {"name": "extended_recent", "extended": True, "decay": 0.55,
     "iterations": 220, "seed_offset": 1000},
    {"name": "extended_latest", "extended": True, "decay": 0.30,
     "iterations": 199, "seed_offset": 2000},
)
MODEL_WEIGHTS = np.asarray((0.27358084, 0.26512224, 0.46129691), dtype=float)
R_SCALE = 0.15
SCALE_AUDIT = (0.05, 0.10, 0.15, 0.20, 0.25)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--seeds", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def parameters(
    args: argparse.Namespace, seed: int, iterations: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "iterations": iterations,
        "depth": 8,
        "learning_rate": 0.035,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 12.0,
        "random_strength": 0.35,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "one_hot_max_size": 16,
        "random_seed": seed,
        "task_type": args.task_type,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if args.task_type == "GPU":
        result.update(devices=args.devices, border_count=32, gpu_ram_part=0.85)
    return result


def train_channel(
    features: pd.DataFrame,
    hierarchy: np.ndarray,
    target: np.ndarray,
    seasons: np.ndarray,
    validation_year: int,
    spec: dict[str, object],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    train = seasons < validation_year
    valid = seasons == validation_year
    sample_weight = np.power(
        float(spec["decay"]), (validation_year - 1) - seasons[train],
    )
    residual = target[train] - hierarchy[train]
    members: list[np.ndarray] = []
    audits: list[dict[str, float]] = []
    for seed in SEEDS[:args.seeds]:
        actual_seed = int(seed + int(spec["seed_offset"]))
        model = CatBoostRegressor(**parameters(
            args, actual_seed, int(spec["iterations"]),
        ))
        model.fit(
            features.loc[train], residual,
            sample_weight=sample_weight,
            cat_features=list(BASE_CATEGORICAL),
        )
        prediction = np.clip(
            hierarchy[valid] + model.predict(features.loc[valid]), *CLIP,
        )
        members.append(prediction)
        audits.append({
            "seed": actual_seed,
            "score": float(bss(target[valid], prediction)),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
        })
        del model
        gc.collect()
    return np.mean(members, axis=0), audits


def fit_fold(
    raw_all: pd.DataFrame,
    validation_year: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    raw = raw_all.loc[raw_all["season"].le(validation_year)].reset_index(drop=True)
    target = raw[TARGET_COL].to_numpy(np.float32)
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < validation_year
    prior = float(target[train].mean())
    snapshots = build_snapshots(raw.loc[train])
    channel_predictions: list[np.ndarray] = []
    channel_audits: dict[str, object] = {}

    # Build each representation once, train all channels that use it, then free
    # it before constructing the other matrix.  This keeps peak RAM bounded.
    for extended in (False, True):
        features, hierarchy = build_features(
            raw, prior, snapshots, extended=extended,
        )
        for spec in MODEL_SPECS:
            if bool(spec["extended"]) != extended:
                continue
            prediction, audits = train_channel(
                features, hierarchy, target, seasons, validation_year,
                spec, args,
            )
            channel_predictions.append(prediction)
            channel_audits[str(spec["name"])] = {
                "decay": float(spec["decay"]),
                "iterations": int(spec["iterations"]),
                "seed_members": audits,
                "ensemble_score": float(bss(target[seasons == validation_year], prediction)),
            }
        del features, hierarchy
        gc.collect()

    ensemble = np.average(
        np.stack(channel_predictions, axis=0), axis=0, weights=MODEL_WEIGHTS,
    )
    return ensemble, {
        "prior": prior,
        "channels": channel_audits,
        "ensemble_score": float(bss(target[seasons == validation_year], ensemble)),
    }


def gain(target: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(target, candidate) - bss(target, anchor))


def report_segments(
    target: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    regular: np.ndarray,
) -> dict[str, object]:
    positions = np.arange(len(target))
    halves = np.array_split(positions, 2)
    quarters = np.array_split(positions, 4)
    return {
        "gain": gain(target, anchor, candidate),
        "half_gains": [gain(target[p], anchor[p], candidate[p]) for p in halves],
        "quarter_gains": [gain(target[p], anchor[p], candidate[p]) for p in quarters],
        "regular_gain": (
            gain(target[regular], anchor[regular], candidate[regular])
            if regular.any() else None
        ),
        "futures_gain": (
            gain(target[~regular], anchor[~regular], candidate[~regular])
            if (~regular).any() else None
        ),
        "mean_absolute_change": float(np.mean(np.abs(candidate - anchor))),
    }


def pitcher_bootstrap(
    target: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(target.mean() * (1.0 - target.mean()))
    row_improvement = np.square(anchor - target) - np.square(candidate - target)
    grouped = pd.DataFrame({
        "pitcher": pitcher.astype(str), "improvement": row_improvement,
    }).groupby("pitcher", sort=False)["improvement"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        sampled = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start:start + count] = (
            100_000.0 * sums[sampled].sum(axis=1)
            / sizes[sampled].sum(axis=1) / reference
        )
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def main() -> None:
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    alternatives: dict[int, np.ndarray] = {}
    fold_audits: dict[str, object] = {}
    for year in (2023, 2024):
        alternatives[year], fold_audits[str(year)] = fit_fold(raw, year, args)

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        oof = {key: archive[key] for key in archive.files}
    target = oof["target"].astype(float)
    season = oof["season"].astype(int)
    anchor = oof["blended"].astype(float)
    alternative = np.concatenate([alternatives[2023], alternatives[2024]])
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(anchor) or not np.array_equal(
        rows[TARGET_COL].to_numpy(float), target,
    ):
        raise ValueError("v64 OOF predictions are not aligned with train.csv")
    regular = rows["game_type"].astype(str).eq("R").to_numpy()

    scale_audit: dict[str, object] = {}
    predictions: dict[float, np.ndarray] = {}
    for scale in SCALE_AUDIT:
        candidate = anchor.copy()
        candidate[regular] += scale * (
            alternative[regular] - anchor[regular]
        )
        candidate = np.clip(candidate, *CLIP)
        predictions[scale] = candidate
        scale_audit[str(scale)] = {
            str(year): report_segments(
                target[season == year], anchor[season == year],
                candidate[season == year], regular[season == year],
            ) for year in (2023, 2024)
        }

    selected = predictions[R_SCALE]
    bootstrap = {
        str(year): pitcher_bootstrap(
            target[(season == year) & regular],
            anchor[(season == year) & regular],
            selected[(season == year) & regular],
            rows.loc[(season == year) & regular, "pitcher_id"].to_numpy(),
            args.bootstrap, 660000 + year,
        ) for year in (2023, 2024)
    }
    selected_reports = scale_audit[str(R_SCALE)]
    strict_gate = bool(
        all(selected_reports[str(year)]["gain"] > 0.0 for year in (2023, 2024))
        and all(bootstrap[str(year)]["positive_probability"] >= 0.80
                for year in (2023, 2024))
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "candidate": "clean_room_hierarchical_residual_seed_ensemble",
        "reference_fixed_r_scale": R_SCALE,
        "f_scale": 0.0,
        "seeds": list(SEEDS[:args.seeds]),
        "model_weights": MODEL_WEIGHTS.tolist(),
        "fold_models": fold_audits,
        "selected_reports": selected_reports,
        "scale_audit": scale_audit,
        "bootstrap": bootstrap,
        "strict_gate": strict_gate,
        "rules": {
            "official_train_only_for_fit": True,
            "external_model_or_prediction_used_for_fit": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "research/v66_hierarchical_residual.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/v66_hierarchical_residual_oof.npz",
        target=target, season=season, anchor=anchor,
        alternative=alternative, blended=selected,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
