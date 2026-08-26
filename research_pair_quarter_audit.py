"""Audit robust pair residual effects on individual forward-time quarters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_pair_residual_tables import apply_pair, fit_pair
from research_residual_portfolio_v19 import prepare


def score_gain(y, base, correction):
    uncertainty = float(y.mean() * (1. - y.mean()))
    residual = y - base
    return float(100000. * np.mean(
        2. * residual * correction - correction * correction
    ) / uncertainty)


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    rows = {year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
            for year in (2023, 2024)}
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        folds[year] = {"y": oof["target"][mask].astype(float),
                       "base": oof["blended"][mask].astype(float)}
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]
    q = {year: np.linspace(0, len(rows[year]), 5, dtype=int) for year in rows}
    sources = {
        "23h1": (rows[2023].iloc[:q[2023][2]], folds[2023]["residual"][:q[2023][2]], folds[2023]["base"][:q[2023][2]]),
        "23": (rows[2023], folds[2023]["residual"], folds[2023]["base"]),
        "24h1": (rows[2024].iloc[:q[2024][2]], folds[2024]["residual"][:q[2024][2]], folds[2024]["base"][:q[2024][2]]),
    }
    blocks = {}
    for index in (2, 3):
        for year, source in ((2023, "23h1"), (2024, "24h1")):
            start, stop = q[year][index], q[year][index + 1]
            blocks[f"{year}_h1_to_q{index + 1}"] = (
                rows[year].iloc[start:stop], folds[year]["y"][start:stop],
                folds[year]["base"][start:stop], source,
            )
    for index in range(4):
        start, stop = q[2024][index], q[2024][index + 1]
        blocks[f"2023_to_2024_q{index + 1}"] = (
            rows[2024].iloc[start:stop], folds[2024]["y"][start:stop],
            folds[2024]["base"][start:stop], "23",
        )
    reports = json.loads(
        (root / "research/pair_residual_tables_v19.json").read_text("utf-8")
    )
    best, audit = {}, []
    for config in reports:
        best.setdefault((config["name"], config["context"]), config)
    for config in best.values():
        if config["min_transfer"] <= .01:
            continue
        pair = tuple(config["features"])
        fitted = {label: fit_pair(frame, residual, base, pair, config["context"],
                                  int(config["n_bins"]), float(config["shrink"]))
                  for label, (frame, residual, base) in sources.items()}
        gains = {}
        for label, (frame, y, base, source) in blocks.items():
            correction = float(config["weight"]) * apply_pair(
                frame, base, pair, config["context"], fitted[source]
            )
            gains[label] = score_gain(y, base, correction)
        audit.append({"name": config["name"], "context": config["context"],
                      "n_bins": config["n_bins"], "shrink": config["shrink"],
                      "weight": config["weight"], "gains": gains,
                      "min_quarter": min(gains.values()),
                      "mean_quarter": float(np.mean(list(gains.values())))})
    audit.sort(key=lambda row: (row["min_quarter"], row["mean_quarter"]), reverse=True)
    output = root / "research/pair_quarter_audit_v19.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
