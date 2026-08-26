"""Screen recency-weighted count specialists against the v14 ensemble.

This is deliberately a diagnostic-only script: it trains one seed for each
candidate on seasons before 2024, predicts 2024, and reports both an optimistic
full-year blend and a five-block cross-fitted blend.  Only candidates improving
the latter should be promoted to a submission version.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)


CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]
V14_COMPONENTS = ["catboost", "count_expert", "categorical_catboost",
                  "categorical_count_expert", "brier_regressor", "weighted_catboost"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--oof", default="outputs/v14_oof_predictions.npz")
    parser.add_argument("--output", default="research/weighted_specialists_2024.npz")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=1200)
    return parser.parse_args()


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
                      / (rate * (1. - rate)))


def recency_weights(seasons, reference):
    age = np.maximum(0., reference - seasons.astype(np.float64))
    return np.exp(-np.log(2.) * age / 3.).astype(np.float32)


def model_params(args, seed, regressor=False):
    result = dict(
        iterations=args.iterations, learning_rate=.02, depth=6,
        loss_function="RMSE" if regressor else "Logloss",
        eval_metric="RMSE" if regressor else "Logloss",
        l2_leaf_reg=100., random_strength=1., random_seed=seed,
        border_count=32, allow_writing_files=False, verbose=0,
        task_type=args.task_type, thread_count=args.threads,
    )
    if args.task_type == "GPU":
        result["devices"] = args.devices
    return result


def train_routed_candidate(features, target, seasons, args, kind):
    train = seasons < 2024
    valid = seasons == 2024
    train_gate = features.loc[train, "two_strike"].to_numpy().astype(bool)
    valid_gate = features.loc[valid, "two_strike"].to_numpy().astype(bool)
    prediction = np.empty(valid.sum(), dtype=np.float64)
    for gate_value in (False, True):
        train_rows = train.copy()
        train_rows[train] = train_gate == gate_value
        valid_rows = valid_gate == gate_value
        is_regressor = kind == "weighted_brier_specialist"
        cls = CatBoostRegressor if is_regressor else CatBoostClassifier
        params = model_params(args, 141 + int(gate_value), is_regressor)
        if kind == "weighted_categorical_specialist":
            params.update(max_ctr_complexity=1, one_hot_max_size=32)
        model = cls(**params)
        fit_kwargs = {}
        if kind == "weighted_categorical_specialist":
            fit_kwargs["cat_features"] = CAT_COLUMNS
        model.fit(
            features.loc[train_rows], target[train_rows],
            sample_weight=recency_weights(seasons[train_rows], 2023),
            **fit_kwargs,
        )
        if is_regressor:
            prediction[valid_rows] = model.predict(features.loc[valid].loc[valid_rows])
        else:
            prediction[valid_rows] = model.predict_proba(
                features.loc[valid].loc[valid_rows]
            )[:, 1]
    return prediction


def fit_segment_blends(target, matrix, gate, train_mask):
    params = {}
    width = matrix.shape[1]
    for label, value in (("other", False), ("two_strike", True)):
        mask = train_mask & (gate == value)
        y, x = target[mask], matrix[mask]
        reference = float(y.mean() * (1. - y.mean()))

        def objective(z):
            prediction = np.clip(z[width] + z[width + 1] * (x @ z[:width]), .005, .995)
            return float(np.mean((y - prediction) ** 2) / reference
                         + .005 * z[width] ** 2 + .002 * (z[width + 1] - 1.) ** 2)

        result = minimize(
            objective, np.r_[np.full(width, 1. / width), 0., 1.], method="SLSQP",
            bounds=[(0., 1.)] * width + [(-.08, .08), (.75, 1.25)],
            constraints={"type": "eq", "fun": lambda z: z[:width].sum() - 1.},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(result.message)
        params[label] = result.x
    return params


def apply_blends(matrix, gate, params):
    prediction = np.empty(len(matrix), dtype=np.float64)
    for label, value in (("other", False), ("two_strike", True)):
        mask = gate == value
        z = params[label]
        prediction[mask] = np.clip(z[-2] + z[-1] * (matrix[mask] @ z[:-2]), .005, .995)
    return prediction


def evaluate(target, matrix, gate):
    all_rows = np.ones(len(target), dtype=bool)
    params = fit_segment_blends(target, matrix, gate, all_rows)
    fitted = apply_blends(matrix, gate, params)
    crossfit = np.empty(len(target), dtype=np.float64)
    for held_indices in np.array_split(np.arange(len(target)), 5):
        train_mask = np.ones(len(target), dtype=bool)
        train_mask[held_indices] = False
        held_params = fit_segment_blends(target, matrix, gate, train_mask)
        held_prediction = apply_blends(matrix[held_indices], gate[held_indices], held_params)
        crossfit[held_indices] = held_prediction
    return bss(target, fitted), bss(target, crossfit), params


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

    names = ["weighted_specialist", "weighted_categorical_specialist", "weighted_brier_specialist"]
    specialist_predictions = []
    for name in names:
        prediction = train_routed_candidate(features, target, seasons, args, name)
        specialist_predictions.append(prediction)
        valid_target = target[seasons == 2024]
        valid_gate = features.loc[seasons == 2024, "two_strike"].to_numpy().astype(bool)
        print(name, {
            "all_bss": bss(valid_target, prediction),
            "other_bss": bss(valid_target[~valid_gate], prediction[~valid_gate]),
            "two_strike_bss": bss(valid_target[valid_gate], prediction[valid_gate]),
        }, flush=True)

    with np.load(args.oof, allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    latest = oof["season"] == 2024
    if not np.allclose(target[seasons == 2024], oof["target"][latest]):
        raise ValueError("v14 OOF and train.csv do not align")
    raw_names = list(oof["model_names"].astype(str))
    indices = [raw_names.index(name) for name in V14_COMPONENTS]
    base = oof["predictions"][latest][:, indices]
    valid_target = oof["target"][latest].astype(np.float64)
    gate = oof["two_strike"][latest].astype(bool)
    specialists = np.column_stack(specialist_predictions)

    configurations = {"v14": base}
    for index, name in enumerate(names):
        configurations[name] = np.column_stack([base, specialists[:, index]])
    configurations["all_specialists"] = np.column_stack([base, specialists])
    reports = {}
    for name, matrix in configurations.items():
        fitted_score, crossfit_score, params = evaluate(valid_target, matrix, gate)
        reports[name] = (fitted_score, crossfit_score)
        weights = {key: np.round(value[:-2], 4).tolist() for key, value in params.items()}
        print(
            f"{name}: fitted={fitted_score:.4f} five_block_cv={crossfit_score:.4f}; "
            f"weights={weights}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, predictions=specialists, names=np.asarray(names),
        target=valid_target.astype(np.float32), two_strike=gate,
        reports=np.asarray([[*reports[name]] for name in configurations]),
        report_names=np.asarray(list(configurations)),
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
