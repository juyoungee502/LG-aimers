"""Train the selected classical ensemble and save submit/model/model_bundle.pkl."""

from dataclasses import asdict
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    ID_COL, TARGET_COL, build_end_history, engineer_features,
    training_history_arrays,
)
from train_submission import make_weights, model_params


LGB_FULL_TREES = 216
LGB_RECENT_TREES = 171
CATBOOST_TREES = 228


def main() -> None:
    test_columns = list(pd.read_csv("./data/test.csv", nrows=0).columns)
    feature_columns = [c for c in test_columns if c != ID_COL]
    train = pd.read_csv(
        "./data/train.csv", usecols=feature_columns + [TARGET_COL]
    )
    target = train.pop(TARGET_COL).astype(np.float32)
    raw = train

    print("Building selected 111 time-safe features...")
    started = time.time()
    bases = training_history_arrays(raw, target)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    print("Features:", features.shape, "seconds:", time.time() - started)

    print("Training full-history LightGBM...")
    lgb_full = lgb.LGBMClassifier(**model_params(42, 0, LGB_FULL_TREES))
    lgb_full.fit(features, target, sample_weight=make_weights(raw["season"]))

    print("Training 2024-only LightGBM...")
    recent = raw["season"].to_numpy() == 2024
    lgb_recent = lgb.LGBMClassifier(
        **model_params(2103, 0, LGB_RECENT_TREES)
    )
    lgb_recent.fit(features.loc[recent], target.loc[recent])

    categorical = [
        "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "pitcher_hand", "batter_hand", "count_state", "hand_matchup",
        "team_matchup",
    ]
    for col in categorical:
        features[col] = features[col].fillna(-1).astype(np.int32)

    print("Training full-history CatBoost...")
    catboost = CatBoostClassifier(
        iterations=CATBOOST_TREES,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        l2_leaf_reg=12.0,
        random_strength=0.6,
        random_seed=2026,
        border_count=96,
        one_hot_max_size=16,
        thread_count=6,
        allow_writing_files=False,
        verbose=50,
    )
    catboost.fit(
        features, target,
        cat_features=categorical,
        sample_weight=make_weights(raw["season"]),
    )

    history = asdict(build_end_history(raw, target))
    bundle = {
        "version": "v2_lgb_cat_season_reconstruction",
        "models": {
            "lgb_full": lgb_full,
            "lgb_recent": lgb_recent,
            "catboost": catboost,
        },
        "blend_weights": {
            "lgb_full": 0.404,
            "lgb_recent": 0.189,
            "catboost": 0.407,
        },
        # 75% of the 2024 OOF affine correction. It is fixed from train only.
        "calibration_intercept": -0.042251175,
        "calibration_slope": 1.06974445,
        "history": history,
        "feature_columns": list(features.columns),
        "cat_features": categorical,
        "clip": [0.01, 0.99],
        "training_info": {
            "lgb_full_trees": LGB_FULL_TREES,
            "lgb_recent_trees": LGB_RECENT_TREES,
            "catboost_trees": CATBOOST_TREES,
            "validation_brier": 0.2476676669,
            "validation_score": 856.36,
        },
    }
    os.makedirs("./submit/model", exist_ok=True)
    path = "./submit/model/model_bundle.pkl"
    joblib.dump(bundle, path, compress=3)
    print("Saved", path, f"{os.path.getsize(path) / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
