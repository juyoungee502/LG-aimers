"""Screen post-2022 Futures-league specialists against the current ensemble.

The F rows have a sharp target-definition/distribution break in 2023.  This
diagnostic deliberately validates only the stable post-break transfer
(train 2023, validate 2024) and reports two chronological halves.  It does not
modify submission artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)


ID_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "team_matchup",
]
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--oof", default="outputs/v17_oof_predictions.npz")
    parser.add_argument("--output", default="research/f_regime_2024.npz")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def params(args, seed, depth, rmse=False, categorical=False):
    result = dict(
        iterations=args.iterations, learning_rate=.025, depth=depth,
        loss_function="RMSE" if rmse else "Logloss",
        eval_metric="RMSE" if rmse else "Logloss",
        l2_leaf_reg=100., random_strength=1., random_seed=seed,
        border_count=32, allow_writing_files=False, verbose=0,
        task_type=args.task_type, thread_count=args.threads,
    )
    if categorical:
        result.update(max_ctr_complexity=1, one_hot_max_size=32)
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def predict_candidate(
    name, features, target, train_mask, valid_mask, args,
    *, depth=6, no_ids=False, categorical=False, rmse=False,
):
    columns = [column for column in features if not (no_ids and column in ID_COLUMNS)]
    cls = CatBoostRegressor if rmse else CatBoostClassifier
    model = cls(**params(args, 1800 + len(name), depth, rmse, categorical))
    fit_kwargs = {}
    if categorical:
        fit_kwargs["cat_features"] = [column for column in CAT_COLUMNS if column in columns]
    model.fit(features.loc[train_mask, columns], target[train_mask], **fit_kwargs)
    if rmse:
        return model.predict(features.loc[valid_mask, columns])
    return model.predict_proba(features.loc[valid_mask, columns])[:, 1]


def best_blend(target, base, candidate, mode):
    if mode == "linear":
        combine = lambda w: (1. - w) * base + w * candidate
    else:
        eps = 1e-5
        base_logit = np.log(np.clip(base, eps, 1-eps) / np.clip(1-base, eps, 1-eps))
        candidate_logit = np.log(
            np.clip(candidate, eps, 1-eps) / np.clip(1-candidate, eps, 1-eps)
        )
        combine = lambda w: 1. / (1. + np.exp(-((1. - w) * base_logit + w * candidate_logit)))
    choices = [(bss(target, combine(weight)), weight) for weight in np.arange(0., .601, .025)]
    score, weight = max(choices)
    return score, weight, combine(weight)


def main():
    args = arguments()
    raw = pd.read_csv(Path(args.data_dir) / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)

    seasons = raw["season"].to_numpy(np.int16)
    f_gate = raw["game_type"].eq("F").to_numpy()
    valid = (seasons == 2024) & f_gate
    train_f = (seasons == 2023) & f_gate
    train_all = seasons == 2023

    with np.load(args.oof) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    latest = oof["season"] == 2024
    latest_rows = raw.loc[seasons == 2024]
    latest_f = latest_rows["game_type"].eq("F").to_numpy()
    if not np.allclose(target[valid], oof["target"][latest][latest_f]):
        raise ValueError("OOF rows do not align with train.csv")
    base = oof["blended"][latest][latest_f].astype(np.float64)
    valid_target = target[valid].astype(np.float64)
    print(
        f"F post-break fold: train_f={train_f.sum()}, train_all={train_all.sum()}, "
        f"valid={valid.sum()}, base_bss={bss(valid_target, base):.6f}", flush=True,
    )

    specs = [
        ("f_numeric_d4", train_f, dict(depth=4)),
        ("f_numeric_d6", train_f, dict(depth=6)),
        ("f_no_ids_d4", train_f, dict(depth=4, no_ids=True)),
        ("f_no_ids_d6", train_f, dict(depth=6, no_ids=True)),
        ("f_categorical_d6", train_f, dict(depth=6, categorical=True)),
        ("f_no_ids_rmse_d6", train_f, dict(depth=6, no_ids=True, rmse=True)),
        ("all_recent_no_ids_d6", train_all, dict(depth=6, no_ids=True)),
    ]
    predictions = []
    reports = []
    base_score = bss(valid_target, base)
    midpoint = len(valid_target) // 2
    halves = [np.arange(len(valid_target)) < midpoint, np.arange(len(valid_target)) >= midpoint]
    for name, train_mask, options in specs:
        prediction = predict_candidate(
            name, features, target, train_mask, valid, args, **options,
        )
        predictions.append(prediction)
        candidates = []
        for mode in ("linear", "logit"):
            score, weight, blended = best_blend(valid_target, base, prediction, mode)
            half_gains = [
                bss(valid_target[mask], blended[mask]) - bss(valid_target[mask], base[mask])
                for mask in halves
            ]
            candidates.append((min(half_gains), score, mode, weight, half_gains))
        robust, score, mode, weight, half_gains = max(candidates)
        report = [bss(valid_target, prediction), score, weight, *half_gains]
        reports.append(report)
        print(
            f"{name}: solo={report[0]:.6f}; {mode} w={weight:.3f}; "
            f"blend={score:.6f} gain={score-base_score:+.6f}; "
            f"half_gains={half_gains}; robust={robust:+.6f}; mean={prediction.mean():.6f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, names=np.asarray([spec[0] for spec in specs]),
        predictions=np.column_stack(predictions), target=valid_target.astype(np.float32),
        base=base, reports=np.asarray(reports),
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
