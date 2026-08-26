"""Measure low-cardinality residual-table gains after v20."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_residual_portfolio_v19 import SPECS, apply_table, build_table, prepare
from residual_portfolio import apply_frozen_portfolio
from train_residual_portfolio import freeze as freeze_v20


def gain(y, base, correction):
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
    old_configs = {label: freeze_v20(frame, residual)
                   for label, (frame, residual, _base) in sources.items()}
    old = {label: apply_frozen_portfolio(frame, old_configs[source])
           for label, (frame, _y, _base, source) in blocks.items()}
    old_gains = {label: gain(y, base, old[label])
                 for label, (_frame, y, base, _source) in blocks.items()}

    reports = json.loads(
        (root / "research/residual_portfolio_v19.json").read_text("utf-8")
    )
    best, audits = {}, []
    for config in reports:
        best.setdefault(config["name"], config)
    for name, config in best.items():
        if config["min_transfer"] <= .01 or name == "batter_phand":
            continue
        keys = SPECS[name]
        tables = {label: build_table(frame, residual, keys, float(config["shrink"]))
                  for label, (frame, residual, _base) in sources.items()}
        values = {label: float(config["weight"]) * np.where(
            frame["game_type"].eq("R").to_numpy(),
            apply_table(frame, tables[source], keys), 0.,
        ) for label, (frame, _y, _base, source) in blocks.items()}
        for scale in (.25, .5, .75, 1., 1.25):
            marginal, total = {}, {}
            for label, (_frame, y, base, _source) in blocks.items():
                total[label] = gain(y, base, old[label] + scale * values[label])
                marginal[label] = total[label] - old_gains[label]
            audits.append({"name": name, "keys": keys, "shrink": config["shrink"],
                           "base_weight": config["weight"], "scale": scale,
                           "marginal_gains": marginal, "total_gains": total,
                           "min_marginal": min(marginal.values()),
                           "mean_marginal": float(np.mean(list(marginal.values())))})
    audits.sort(key=lambda row: (row["min_marginal"], row["mean_marginal"]), reverse=True)
    result = {"v20_gains": old_gains, "audits": audits}
    output = root / "research/v20_lowcard_additions.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"v20_gains": old_gains, "top": audits[:30]}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
