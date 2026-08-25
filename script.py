"""Evaluation entry point; writes output/submission.csv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import ID_COL, TARGET_COL, build_features

def main():
    data, models, output = Path("data"), Path("model"), Path("output")
    test = pd.read_csv(data / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig")
    meta = json.loads((models / "metadata.json").read_text(encoding="utf-8"))
    if not test[ID_COL].is_unique: raise ValueError("test row_id must be unique")
    if set(test[ID_COL]) != set(sample[ID_COL]): raise ValueError("row_id sets differ")
    x = build_features(test); expected = meta["feature_columns"]
    missing = sorted(set(expected) - set(x.columns))
    if missing: raise ValueError(f"Missing features: {missing}")
    model = CatBoostClassifier(); model.load_model(str(models / "catboost.cbm"))
    p = model.predict_proba(x[expected])[:, 1]; cal = meta["calibration"]
    if cal["strength"] > 0:
        logit = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
        z = (1-cal["strength"])*logit + cal["strength"]*(cal["a"]*logit+cal["b"])
        p = 1/(1+np.exp(-np.clip(z, -30, 30)))
    pred = pd.Series(p, index=test[ID_COL]).reindex(sample[ID_COL]).to_numpy()
    if not np.isfinite(pred).all(): raise ValueError("Non-finite predictions")
    sample[TARGET_COL] = np.clip(pred, 0, 1); output.mkdir(exist_ok=True)
    sample[[ID_COL, TARGET_COL]].to_csv(output / "submission.csv", index=False)
    print(f"Saved {len(sample):,} predictions")

if __name__ == "__main__": main()
