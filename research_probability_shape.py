"""Test train-only nonlinear calibration of the v19 probability shape."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def design(probability, method):
    p = np.clip(probability, 1e-5, 1. - 1e-5)
    if method == "affine":
        return p[:, None]
    if method == "quadratic":
        return np.column_stack([p, (p - .5) ** 2])
    if method == "cubic":
        return np.column_stack([p, (p - .5) ** 2, (p - .5) ** 3])
    if method == "platt":
        return logit(p)[:, None]
    if method == "beta":
        return np.column_stack([np.log(p), -np.log1p(-p)])
    raise ValueError(method)


def fit_transform(train_p, train_y, test_p, method):
    if method == "isotonic":
        # Equal-frequency binning limits the variance of unconstrained isotonic
        # calibration while preserving a flexible monotone probability shape.
        order = np.argsort(train_p)
        chunks = np.array_split(order, 200)
        x = np.asarray([train_p[index].mean() for index in chunks])
        y = np.asarray([train_y[index].mean() for index in chunks])
        model = IsotonicRegression(y_min=.005, y_max=.995, out_of_bounds="clip")
        model.fit(x, y)
        return model.predict(test_p)
    x_train = design(train_p, method)
    x_test = design(test_p, method)
    if method in ("platt", "beta"):
        model = LogisticRegression(C=1e4, max_iter=1000)
        model.fit(x_train, train_y)
        return model.predict_proba(x_test)[:, 1]
    model = Ridge(alpha=10.)
    model.fit(x_train, train_y)
    return np.clip(model.predict(x_test), .005, .995)


def main():
    root = Path(__file__).resolve().parent
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    data = pd.read_csv(
        root / "data/train.csv", usecols=["season", "game_type", "strikes_before"],
        encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([
        data.loc[data["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    y = oof["target"].astype(np.float64)
    p = oof["blended"].astype(np.float64)
    if len(rows) != len(y):
        raise ValueError("v19 OOF rows do not align")
    years = oof["season"]
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "R": rows["game_type"].eq("R").to_numpy(),
        "F": rows["game_type"].eq("F").to_numpy(),
        "R_two_strike": (
            rows["game_type"].eq("R") & rows["strikes_before"].eq(2)
        ).to_numpy(),
        "R_other": (
            rows["game_type"].eq("R") & rows["strikes_before"].ne(2)
        ).to_numpy(),
    }
    reports = []
    for segment, active in masks.items():
        # The all transform is used for every row. Segment transforms leave
        # other rows exactly at the v19 baseline.
        source_segment = active if segment != "all" else np.ones(len(y), dtype=bool)
        for method in ("affine", "quadratic", "cubic", "platt", "beta", "isotonic"):
            checks = []
            for name, train_mask, valid_mask in (
                (
                    "2023_h2",
                    (years == 2023) & (np.arange(len(y)) < np.flatnonzero(years == 2023)[len(np.flatnonzero(years == 2023)) // 2]),
                    (years == 2023) & (np.arange(len(y)) >= np.flatnonzero(years == 2023)[len(np.flatnonzero(years == 2023)) // 2]),
                ),
                ("2024", years == 2023, years == 2024),
            ):
                fit = train_mask & source_segment
                apply = valid_mask & source_segment
                prediction = p[valid_mask].copy()
                local_apply = source_segment[valid_mask]
                prediction[local_apply] = fit_transform(
                    p[fit], y[fit], p[apply], method,
                )
                target = y[valid_mask]
                base = p[valid_mask]
                midpoint = len(target) // 2
                checks.append({
                    "name": name,
                    "gain": bss(target, prediction) - bss(target, base),
                    "gain_first_half": bss(target[:midpoint], prediction[:midpoint]) - bss(target[:midpoint], base[:midpoint]),
                    "gain_second_half": bss(target[midpoint:], prediction[midpoint:]) - bss(target[midpoint:], base[midpoint:]),
                    "mean": float(prediction.mean()),
                })
            reports.append({
                "segment": segment, "method": method, "checks": checks,
                "min_gain": min(check["gain"] for check in checks),
                "min_slice": min(
                    min(check["gain_first_half"], check["gain_second_half"])
                    for check in checks
                ),
            })
    reports.sort(
        key=lambda row: (row["min_gain"], row["min_slice"], row["checks"][-1]["gain"]),
        reverse=True,
    )
    output = root / "research/probability_shape_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:40], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
