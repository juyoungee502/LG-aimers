"""Strict-forward audit of a complete hierarchical residual stack.

This clean-room experiment adds three pieces that the earlier v66 screen did
not contain: Futures-regime regressors, reconstructed failure-cause risks, and
an adaptive residual gate.  Each validation season is predicted only from
earlier seasons; the gate itself is also trained only on earlier OOF folds.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from research_inferred_pitch_priors import reconstruct_labels
from research_v66_hierarchical_residual import (
    MODEL_SPECS, MODEL_WEIGHTS, SEEDS, gain, parameters, pitcher_bootstrap,
    report_segments,
)
from train_v25_temporal_portfolio import bss
from v66_hierarchical_residual import (
    BASE_CATEGORICAL, CLIP, TARGET_COL, build_adaptive_gate_features,
    build_features, build_snapshots,
)


ROOT = Path(__file__).resolve().parent
STACK_INTERCEPT = 0.0300329767
STACK_COEFFICIENTS = np.asarray(
    (0.93505266, -0.00520129, 0.01091677, -0.02528331), dtype=float,
)
FAILURE_NAMES = ("middle", "wayoff", "reverse")
FAILURE_ITERATIONS = (100, 190, 230)
REFERENCE_R_SCALE = 0.25
REFERENCE_F_SCALE = 0.10
SCALE_GRID = (
    (0.10, 0.00), (0.15, 0.025), (0.20, 0.05),
    (0.25, 0.075), (0.25, 0.10), (0.30, 0.10),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--main-seeds", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--f-seed-factor", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def fit_regression_members(
    features: pd.DataFrame,
    base: np.ndarray,
    target: np.ndarray,
    seasons: np.ndarray,
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
    *,
    decay: float | None,
    iterations: int,
    seeds: list[int],
    args: argparse.Namespace,
    l2: float,
) -> np.ndarray:
    if not fit_mask.any():
        raise ValueError("empty specialist training partition")
    weights = None
    if decay is not None:
        latest = int(seasons[fit_mask].max())
        weights = np.power(decay, latest - seasons[fit_mask])
    members: list[np.ndarray] = []
    for seed in seeds:
        config = parameters(args, seed, iterations)
        config.update(l2_leaf_reg=l2, verbose=False)
        model = CatBoostRegressor(**config)
        model.fit(
            features.loc[fit_mask], target[fit_mask] - base[fit_mask],
            sample_weight=weights, cat_features=list(BASE_CATEGORICAL),
        )
        members.append(np.clip(
            base[predict_mask] + model.predict(features.loc[predict_mask]),
            *CLIP,
        ))
        del model
        gc.collect()
    return np.mean(members, axis=0)


def fit_main_channel(
    features: pd.DataFrame,
    base: np.ndarray,
    target: np.ndarray,
    seasons: np.ndarray,
    validation_year: int,
    spec: dict[str, object],
    args: argparse.Namespace,
) -> np.ndarray:
    train = seasons < validation_year
    valid = seasons == validation_year
    seed_values = [
        int(seed + int(spec["seed_offset"]))
        for seed in SEEDS[:args.main_seeds]
    ]
    return fit_regression_members(
        features, base, target, seasons, train, valid,
        decay=float(spec["decay"]), iterations=int(spec["iterations"]),
        seeds=seed_values, args=args, l2=12.0,
    )


def count_from_factor(base_count: int, factor: float) -> int:
    return max(1, int(round(base_count * factor)))


def replace_futures_channels(
    channels: list[np.ndarray],
    compact: pd.DataFrame,
    compact_base: np.ndarray,
    extended: pd.DataFrame,
    extended_base: np.ndarray,
    target: np.ndarray,
    seasons: np.ndarray,
    game_type: np.ndarray,
    validation_year: int,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    valid = seasons == validation_year
    valid_f = valid & (game_type == "F")
    if not valid_f.any():
        return channels
    train_f = (seasons < validation_year) & (game_type == "F")
    recent_f = train_f & (seasons == validation_year - 1)
    specifications = (
        (compact, compact_base, train_f, 0.55, 140, 4, 668000),
        (extended, extended_base, recent_f, None, 220, 6, 668100),
        (extended, extended_base, train_f, 0.30, 199, 4, 668200),
        (extended, extended_base, recent_f, None, 199, 2, 668300),
    )
    specialists: list[np.ndarray] = []
    for feature, base, fit_mask, decay, iterations, members, seed_start in specifications:
        count = count_from_factor(members, args.f_seed_factor)
        specialists.append(fit_regression_members(
            feature, base, target, seasons, fit_mask, valid_f,
            decay=decay, iterations=iterations,
            seeds=[seed_start + index for index in range(count)],
            args=args, l2=20.0,
        ))

    local_f = game_type[valid] == "F"
    result = [prediction.copy() for prediction in channels]
    result[0][local_f] += 2.0 * (specialists[0] - result[0][local_f])
    result[1][local_f] += 0.5 * (specialists[1] - result[1][local_f])
    recent_inner = result[2][local_f] + 0.25 * (
        specialists[3] - result[2][local_f]
    )
    fused_latest = 0.25 * specialists[2] + 0.75 * recent_inner
    result[2][local_f] += 0.5 * (fused_latest - result[2][local_f])
    return [np.clip(prediction, *CLIP) for prediction in result]


def classifier_configuration(
    args: argparse.Namespace, seed: int, iterations: int, l2: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "iterations": iterations,
        "depth": 7,
        "learning_rate": 0.04,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "l2_leaf_reg": l2,
        "random_strength": 0.4,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "one_hot_max_size": 16,
        "random_seed": seed,
        "task_type": args.task_type,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }
    if args.task_type == "GPU":
        result.update(devices=args.devices, border_count=32, gpu_ram_part=0.85)
    return result


def predict_failure_risks(
    extended: pd.DataFrame,
    labels: pd.DataFrame,
    seasons: np.ndarray,
    game_type: np.ndarray,
    validation_year: int,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    valid = seasons == validation_year
    recovered = labels[list(FAILURE_NAMES)].notna().all(axis=1).to_numpy()
    general_fit = (seasons < validation_year) & recovered
    f_fit = general_fit & (game_type == "F")
    full_valid_f = valid & (game_type == "F")
    local_valid_f = game_type[valid] == "F"
    recent_year = validation_year - 1
    general_weight = np.power(0.30, recent_year - seasons[general_fit])
    f_weight = np.power(0.30, recent_year - seasons[f_fit])
    risks: list[np.ndarray] = []
    for label_index, (name, iterations) in enumerate(zip(
        FAILURE_NAMES, FAILURE_ITERATIONS,
    )):
        label_values = labels[name].to_numpy(float)
        members: list[np.ndarray] = []
        for seed in SEEDS[:args.main_seeds]:
            model = CatBoostClassifier(**classifier_configuration(
                args, 669000 + 100 * label_index + seed, iterations, 12.0,
            ))
            model.fit(
                extended.loc[general_fit], label_values[general_fit],
                sample_weight=general_weight,
                cat_features=list(BASE_CATEGORICAL),
            )
            members.append(model.predict_proba(extended.loc[valid])[:, 1])
            del model
            gc.collect()
        risk = np.mean(members, axis=0)
        if full_valid_f.any() and f_fit.any():
            f_model = CatBoostClassifier(**classifier_configuration(
                args, 669500 + label_index, iterations, 20.0,
            ))
            f_model.fit(
                extended.loc[f_fit], label_values[f_fit],
                sample_weight=f_weight, cat_features=list(BASE_CATEGORICAL),
            )
            f_risk = f_model.predict_proba(extended.loc[full_valid_f])[:, 1]
            risk[local_valid_f] += 0.75 * (
                f_risk - risk[local_valid_f]
            )
            del f_model
            gc.collect()
        risks.append(np.clip(risk, 1e-5, 1.0 - 1e-5))
    return risks


def fit_complete_fold(
    raw_all: pd.DataFrame,
    validation_year: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    raw = raw_all.loc[raw_all["season"].le(validation_year)].reset_index(drop=True)
    target = raw[TARGET_COL].to_numpy(np.float32)
    seasons = raw["season"].to_numpy(np.int16)
    game_type = raw["game_type"].astype(str).to_numpy()
    train = seasons < validation_year
    valid = seasons == validation_year
    prior = float(target[train].mean())
    snapshots = build_snapshots(raw.loc[train])
    labels = reconstruct_labels(raw).reset_index(drop=True)

    compact, compact_base = build_features(raw, prior, snapshots, extended=False)
    channels = [fit_main_channel(
        compact, compact_base, target, seasons, validation_year,
        MODEL_SPECS[0], args,
    )]
    extended, extended_base = build_features(raw, prior, snapshots, extended=True)
    for spec in MODEL_SPECS[1:]:
        channels.append(fit_main_channel(
            extended, extended_base, target, seasons, validation_year, spec, args,
        ))
    channels = replace_futures_channels(
        channels, compact, compact_base, extended, extended_base, target,
        seasons, game_type, validation_year, args,
    )
    risks = predict_failure_risks(
        extended, labels, seasons, game_type, validation_year, args,
    )
    main = np.average(np.stack(channels), axis=0, weights=MODEL_WEIGHTS)
    design = np.column_stack([main, *risks])
    stacked = np.clip(
        STACK_INTERCEPT + design @ STACK_COEFFICIENTS, *CLIP,
    )
    validation_rows = raw.loc[valid].reset_index(drop=True)
    gate_features = build_adaptive_gate_features(
        validation_rows, channels, risks, stacked,
    )
    fold = {
        "year": validation_year,
        "target": target[valid].astype(float),
        "rows": validation_rows,
        "channels": channels,
        "risks": risks,
        "stacked": stacked,
        "gate_features": gate_features,
        "audit": {
            "prior": prior,
            "rows": int(valid.sum()),
            "f_rows": int((game_type[valid] == "F").sum()),
            "channel_scores": [
                float(bss(target[valid], prediction)) for prediction in channels
            ],
            "stacked_score": float(bss(target[valid], stacked)),
        },
    }
    del compact, extended, compact_base, extended_base
    gc.collect()
    return fold


def fit_forward_gate(
    folds: dict[int, dict[str, object]],
    validation_year: int,
    args: argparse.Namespace,
) -> np.ndarray:
    source_years = [year for year in folds if year < validation_year]
    features = pd.concat([
        folds[year]["gate_features"] for year in source_years
    ], ignore_index=True)
    residual = np.concatenate([
        folds[year]["target"] - folds[year]["stacked"]
        for year in source_years
    ])
    weights = np.concatenate([
        np.full(
            len(folds[year]["target"]),
            0.55 ** ((validation_year - 1) - year), dtype=float,
        ) for year in source_years
    ])
    gate_configuration: dict[str, object] = dict(
        iterations=73, depth=3, learning_rate=0.025,
        loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=30.0,
        random_strength=0.2, bootstrap_type="Bernoulli", subsample=0.8,
        random_seed=670000 + validation_year, task_type=args.task_type,
        thread_count=-1, allow_writing_files=False, verbose=False,
    )
    if args.task_type == "GPU":
        gate_configuration.update(devices=args.devices, border_count=32)
    model = CatBoostRegressor(**gate_configuration)
    model.fit(features, residual, sample_weight=weights)
    prediction = np.clip(
        folds[validation_year]["stacked"]
        + model.predict(folds[validation_year]["gate_features"]),
        *CLIP,
    )
    del model
    gc.collect()
    return prediction


def blend_by_regime(
    anchor: np.ndarray,
    alternative: np.ndarray,
    regular: np.ndarray,
    r_scale: float,
    f_scale: float,
) -> np.ndarray:
    scale = np.where(regular, r_scale, f_scale)
    return np.clip(anchor + scale * (alternative - anchor), *CLIP)


def main() -> None:
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    folds: dict[int, dict[str, object]] = {}
    for year in (2022, 2023, 2024):
        print(f"[complete hierarchy] fitting strict fold {year}", flush=True)
        folds[year] = fit_complete_fold(raw, year, args)
        print(json.dumps(folds[year]["audit"], indent=2), flush=True)
    adaptive = {
        year: fit_forward_gate(folds, year, args) for year in (2023, 2024)
    }

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        oof = {key: archive[key] for key in archive.files}
    target = oof["target"].astype(float)
    season = oof["season"].astype(int)
    anchor = oof["blended"].astype(float)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(anchor) or not np.array_equal(
        rows[TARGET_COL].to_numpy(float), target,
    ):
        raise ValueError("v64 OOF predictions are not aligned with train.csv")
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    alternatives = {
        "stacked": np.concatenate([folds[year]["stacked"] for year in (2023, 2024)]),
        "adaptive": np.concatenate([adaptive[year] for year in (2023, 2024)]),
    }

    scale_audit: dict[str, object] = {}
    for name, alternative in alternatives.items():
        scale_audit[name] = {}
        for r_scale, f_scale in SCALE_GRID:
            candidate = blend_by_regime(
                anchor, alternative, regular, r_scale, f_scale,
            )
            scale_audit[name][f"R{r_scale:.3f}_F{f_scale:.3f}"] = {
                str(year): report_segments(
                    target[season == year], anchor[season == year],
                    candidate[season == year], regular[season == year],
                ) for year in (2023, 2024)
            }

    selected = blend_by_regime(
        anchor, alternatives["adaptive"], regular,
        REFERENCE_R_SCALE, REFERENCE_F_SCALE,
    )
    selected_reports = {
        str(year): report_segments(
            target[season == year], anchor[season == year],
            selected[season == year], regular[season == year],
        ) for year in (2023, 2024)
    }
    bootstrap = {
        str(year): pitcher_bootstrap(
            target[season == year], anchor[season == year],
            selected[season == year], rows.loc[
                season == year, "pitcher_id"
            ].to_numpy(), args.bootstrap, 671000 + year,
        ) for year in (2023, 2024)
    }
    strict_gate = bool(
        all(selected_reports[str(year)]["gain"] > 0 for year in (2023, 2024))
        and all(bootstrap[str(year)]["positive_probability"] >= 0.80
                for year in (2023, 2024))
    )
    report = {
        "baseline": "v64_public_1135_1_anchor",
        "candidate": "complete_hierarchy_f_specialists_failure_risks_forward_gate",
        "reference_scales": {
            "regular": REFERENCE_R_SCALE, "futures": REFERENCE_F_SCALE,
        },
        "folds": {str(year): folds[year]["audit"] for year in folds},
        "selected_reports": selected_reports,
        "bootstrap": bootstrap,
        "scale_audit": scale_audit,
        "strict_gate": strict_gate,
        "rules": {
            "official_train_only_for_fit": True,
            "external_weights_or_predictions_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "research/v66_complete_hierarchy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/v66_complete_hierarchy_oof.npz",
        target=target, season=season, anchor=anchor,
        stacked=alternatives["stacked"], adaptive=alternatives["adaptive"],
        blended=selected,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
