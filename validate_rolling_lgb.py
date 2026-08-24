"""Rolling-origin diagnostics used to justify final probability calibration."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_engineering import (
    TARGET_COL, add_training_component_features, engineer_features,
    training_history_arrays,
)
from train_submission import brier_report, make_weights, model_params


def main() -> None:
    cols = list(pd.read_csv("./data/test.csv", nrows=0).columns)
    cols.remove("row_id")
    train = pd.read_csv("./data/train.csv", usecols=cols + [TARGET_COL])
    y = train.pop(TARGET_COL).astype(np.float32)
    bases = training_history_arrays(train, y)
    x = engineer_features(train, *bases, global_prior=float(y.mean()))
    add_training_component_features(x, train)

    reports = []
    for validation_year in (2022, 2023):
        train_mask = train["season"].to_numpy() < validation_year
        valid_mask = train["season"].to_numpy() == validation_year
        model = lgb.LGBMClassifier(**model_params(42, 0, 420))
        model.fit(
            x.loc[train_mask], y.loc[train_mask],
            sample_weight=make_weights(train.loc[train_mask, "season"]),
            eval_set=[(x.loc[valid_mask], y.loc[valid_mask])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(100)],
        )
        pred = model.predict_proba(x.loc[valid_mask])[:, 1]
        actual = y.loc[valid_mask].to_numpy()
        report = brier_report(actual, pred, str(validation_year))
        intercept, slope = np.linalg.lstsq(
            np.column_stack([np.ones(len(pred)), pred]), actual, rcond=None
        )[0]
        report.update({
            "best_iteration": int(model.best_iteration_),
            "oracle_intercept": float(intercept),
            "oracle_slope": float(slope),
            "mean_residual": float(np.mean(actual - pred)),
        })
        reports.append(report)
        np.save(f"./outputs/validation_lgb_{validation_year}_pred.npy", pred)
        np.save(f"./outputs/validation_lgb_{validation_year}_target.npy", actual)

    pd.DataFrame(reports).to_csv("./outputs/validation_rolling_lgb.csv", index=False)
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
