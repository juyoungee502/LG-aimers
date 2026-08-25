"""Train with 2024 temporal validation, calibration, then full-data refit."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from features import CAT_COLUMNS, ID_COL, TARGET_COL, build_features

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--preset", choices=("fast", "full"), default="full")
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
    p.add_argument("--devices", default="0",
                   help="GPU IDs, e.g. 0, 0:1, or 0-3 (ignored on CPU)")
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--seed", type=int, default=20250825)
    return p.parse_args()

def temporal_weights(season, reference, half_life=2.5):
    age = np.maximum(0, reference - season.to_numpy(dtype=np.float32))
    return np.exp(-math.log(2) * age / half_life).astype(np.float32)

def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def auc(y, p):
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def fit_platt(y, p):
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    a, b = 1.0, 0.0
    for _ in range(30):
        q = 1 / (1 + np.exp(-np.clip(a * x + b, -30, 30)))
        w = np.maximum(q * (1 - q), 1e-8)
        g = np.array([np.sum((q-y)*x), np.sum(q-y)])
        h = np.array([[np.sum(w*x*x), np.sum(w*x)], [np.sum(w*x), np.sum(w)]])
        h.flat[::3] += 1e-4
        delta = np.linalg.solve(h, g)
        a, b = (np.array([a, b]) - delta).tolist()
        if np.max(np.abs(delta)) < 1e-7: break
    return float(np.clip(a, .25, 4)), float(np.clip(b, -2, 2))

def calibrate(p, a, b, strength):
    x = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
    z = (1-strength)*x + strength*(a*x+b)
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def model_params(args, iterations):
    d = dict(loss_function="Logloss", eval_metric="Logloss", iterations=iterations,
             depth=8, learning_rate=.055 if args.preset == "full" else .08,
             l2_leaf_reg=8, random_strength=.6, bootstrap_type="Bayesian",
             bagging_temperature=.7, border_count=128, one_hot_max_size=16,
             random_seed=args.seed, task_type=args.task_type, thread_count=args.threads,
             allow_writing_files=False, verbose=100)
    if args.task_type == "CPU":
        d["boosting_type"] = "Plain"
        d["rsm"] = .9
    else:
        d["devices"] = args.devices
    return d

def main():
    args = arguments(); out = Path(args.model_dir); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    if TARGET_COL not in raw or ID_COL not in raw: raise ValueError("Required columns missing")
    y = raw[TARGET_COL].to_numpy(dtype=np.int8); seasons = raw["season"].astype(int)
    X = build_features(raw); cat_idx = [X.columns.get_loc(c) for c in CAT_COLUMNS]
    valid_year = int(seasons.max()); tr, va = seasons < valid_year, seasons == valid_year
    max_iters = 1800 if args.preset == "full" else 500
    vm = CatBoostClassifier(**model_params(args, max_iters))
    vm.fit(Pool(X.loc[tr], y[tr], cat_features=cat_idx,
                weight=temporal_weights(seasons[tr], valid_year-1)),
           eval_set=Pool(X.loc[va], y[va], cat_features=cat_idx),
           early_stopping_rounds=150 if args.preset == "full" else 60, use_best_model=True)
    p = vm.predict_proba(X.loc[va])[:, 1]; a, b = fit_platt(y[va], p)
    choices = [(s, logloss(y[va], calibrate(p, a, b, s))) for s in (0,.25,.5,.75,1)]
    strength, cal_loss = min(choices, key=lambda z: z[1])
    best_iters = max(50, vm.get_best_iteration()+1)
    report = {"validation_year": valid_year, "best_iterations": best_iters,
              "raw_logloss": logloss(y[va], p), "calibrated_logloss": cal_loss,
              "auc": auc(y[va], p), "brier": float(np.mean((y[va]-p)**2)),
              "validation_target_rate": float(y[va].mean()), "raw_prediction_mean": float(p.mean()),
              "platt_a": a, "platt_b": b, "calibration_strength": strength}
    print(json.dumps(report, indent=2)); del vm, p
    fm = CatBoostClassifier(**model_params(args, best_iters))
    fm.fit(Pool(X, y, cat_features=cat_idx,
                weight=temporal_weights(seasons, int(seasons.max()))))
    fm.save_model(str(out / "catboost.cbm"))
    metadata = {"feature_columns": X.columns.tolist(), "categorical_columns": CAT_COLUMNS,
                "calibration": {"a": a, "b": b, "strength": strength}, "report": report}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to {out.resolve()}")

if __name__ == "__main__": main()
