"""Validation-only CatBoost experiment using the shared time-safe features."""

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_training_component_features, engineer_features,
    training_history_arrays,
)
from train_submission import brier_report, make_weights


def main() -> None:
    usecols = list(pd.read_csv("./data/test.csv", nrows=0).columns)
    usecols.remove("row_id")
    train = pd.read_csv("./data/train.csv", usecols=usecols + [TARGET_COL])
    y = train.pop(TARGET_COL).astype(np.float32)
    bases = training_history_arrays(train, y)
    x = engineer_features(train, *bases, global_prior=float(y.mean()))
    add_training_component_features(x, train)

    categorical = [
        "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "pitcher_hand", "batter_hand", "count_state", "hand_matchup",
        "team_matchup",
    ]
    for col in categorical:
        x[col] = x[col].fillna(-1).astype(np.int32)

    valid = train["season"].to_numpy() == 2024
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=12.0,
        random_strength=0.6,
        random_seed=2026,
        border_count=96,
        one_hot_max_size=16,
        thread_count=6,
        allow_writing_files=False,
        verbose=100,
        od_type="Iter",
        od_wait=70,
    )
    started = time.time()
    model.fit(
        x.loc[~valid], y.loc[~valid],
        cat_features=categorical,
        sample_weight=make_weights(train.loc[~valid, "season"]),
        eval_set=(x.loc[valid], y.loc[valid]),
        use_best_model=True,
    )
    pred = model.predict_proba(x.loc[valid])[:, 1]
    report = brier_report(y.loc[valid].to_numpy(), pred, "catboost")
    report["best_iteration"] = model.get_best_iteration()
    report["elapsed_seconds"] = time.time() - started
    print(report)
    pd.DataFrame([report]).to_csv("./outputs/validation_catboost.csv", index=False)
    np.save("./outputs/validation_catboost_pred.npy", pred)


if __name__ == "__main__":
    main()
