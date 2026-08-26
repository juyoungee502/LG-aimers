"""Multi-task CatBoost with training-only pitch labels as auxiliary targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import (
    PITCH_TYPES, bss, reconstruct_labels,
)


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed: int):
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="MultiRMSE",
        eval_metric="MultiRMSE", task_type="GPU", devices="0",
        random_seed=seed, allow_writing_files=False, verbose=0,
    )


def target_matrix(target, labels, variant):
    # Repeating the main target gives it explicit weight while all dimensions
    # still share the same tree structure under MultiRMSE.
    columns = [target] * 4
    columns.extend(labels[name].to_numpy(np.float32) for name in (
        "reverse", "middle", "wayoff", "ball", "strike",
    ))
    names = ["success"] * 4 + ["reverse", "middle", "wayoff", "ball", "strike"]
    if variant == "full_tasks":
        for pitch_type in PITCH_TYPES:
            columns.append(labels["pitch_type"].eq(pitch_type).to_numpy(np.float32))
            names.append(f"pitch_{pitch_type}")
    return np.column_stack(columns).astype(np.float32), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    usable = labels[[
        "reverse", "middle", "wayoff", "ball", "strike", "pitch_type",
    ]].notna().all(axis=1).to_numpy()

    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, prior_season_context(full, labels)], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    seasons = data["season"].to_numpy(np.int16)
    train = (seasons < args.valid_year) & usable
    valid = seasons == args.valid_year

    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    fold = oof["season"] == args.valid_year
    y = oof["target"][fold].astype(np.float64)
    base = oof["blended"][fold].astype(np.float64)
    if not np.allclose(y, target[valid]):
        raise ValueError("v19 OOF rows do not align")
    regular = data.loc[valid, "game_type"].eq("R").to_numpy()
    midpoint = len(y) // 2

    predictions, task_names, reports = {}, {}, []
    for offset, variant in enumerate(("failure_tasks", "full_tasks")):
        targets, names = target_matrix(target, labels, variant)
        model = CatBoostRegressor(**parameters(5100 + offset))
        model.fit(features.loc[train], targets[train])
        matrix = model.predict(features.loc[valid])
        prediction = np.clip(matrix[:, :4].mean(axis=1), .005, .995)
        predictions[variant] = prediction.astype(np.float32)
        task_names[variant] = names
        print(
            f"Multi-task model complete: year={args.valid_year}, "
            f"variant={variant}, dimensions={len(names)}", flush=True,
        )
        for gate_name, active in (
            ("R", regular), ("all", np.ones(len(y), dtype=bool)),
        ):
            for weight in np.arange(-.1, .401, .01):
                blended = base.copy()
                blended[active] = sigmoid(
                    (1. - weight) * logit(base[active])
                    + weight * logit(prediction[active])
                )
                report = {
                    "variant": variant, "gate": gate_name,
                    "weight": float(weight),
                    "gain": bss(y, blended) - bss(y, base),
                    "gain_first_half": bss(y[:midpoint], blended[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                    "gain_second_half": bss(y[midpoint:], blended[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                    "standalone_bss": bss(y[active], prediction[active]),
                    "prediction_mean": float(prediction[active].mean()),
                    "target_mean": float(y[active].mean()),
                }
                report["min_half"] = min(report["gain_first_half"], report["gain_second_half"])
                reports.append(report)
    reports.sort(key=lambda row: (row["min_half"], row["gain"]), reverse=True)
    output = root / f"research/multitask_catboost_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y.astype(np.float32), base=base.astype(np.float32),
        variants=np.asarray(list(predictions)),
        predictions=np.column_stack(list(predictions.values())).astype(np.float32),
        task_names_json=np.asarray(json.dumps(task_names)),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({
        "year": args.valid_year, "usable_train": int(train.sum()),
        "coverage": float(usable.mean()), "top": reports[:40],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
