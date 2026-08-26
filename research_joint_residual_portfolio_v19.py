"""Combine categorical and numeric v19 residual corrections conservatively."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_numeric_residual_tables import apply_effect, fit_effect
from research_residual_portfolio_v19 import apply_table, build_table, prepare


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
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
        }
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]

    m23, m24 = len(rows[2023]) // 2, len(rows[2024]) // 2
    sources = {
        "23h1": (rows[2023].iloc[:m23], folds[2023]["residual"][:m23],
                  folds[2023]["base"][:m23]),
        "23": (rows[2023], folds[2023]["residual"], folds[2023]["base"]),
        "24h1": (rows[2024].iloc[:m24], folds[2024]["residual"][:m24],
                  folds[2024]["base"][:m24]),
    }
    blocks = {
        "2023_h2": (rows[2023].iloc[m23:], folds[2023]["y"][m23:],
                    folds[2023]["base"][m23:], "23h1"),
        "2024_h1": (rows[2024].iloc[:m24], folds[2024]["y"][:m24],
                    folds[2024]["base"][:m24], "23"),
        "2024_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                    folds[2024]["base"][m24:], "23"),
        "2024_h1_to_h2": (rows[2024].iloc[m24:], folds[2024]["y"][m24:],
                           folds[2024]["base"][m24:], "24h1"),
    }

    # Keep one robust setting per feature/context family. The minimum threshold
    # avoids feeding the greedy search thousands of near-zero multiple tests.
    numeric_report = json.loads(
        (root / "research/numeric_residual_tables_v19.json").read_text("utf-8")
    )
    best_by_family = {}
    for row in numeric_report:
        family = (row["feature"], row["context"])
        if family not in best_by_family:
            best_by_family[family] = row
    configs = [row for row in best_by_family.values()
               if row["min_transfer"] >= .03 and abs(row["weight"]) >= .024]

    candidates = {}
    for config in configs:
        name = f"numeric:{config['feature']}:{config['context']}"
        fitted = {}
        for label, (frame, residual, base) in sources.items():
            fitted[label] = fit_effect(
                frame, residual, config["feature"], base, config["context"],
                int(config["n_bins"]), float(config["shrink"]),
            )
        candidates[name] = {}
        for label, (frame, _y, base, source) in blocks.items():
            candidates[name][label] = float(config["weight"]) * apply_effect(
                frame, config["feature"], base, config["context"], fitted[source],
            )

    # Add the ID-based effect found by the separate categorical screen.
    keys, shrink, weight = ["batter_id", "pitcher_hand"], 400., .125
    tables = {
        label: build_table(frame, residual, keys, shrink)
        for label, (frame, residual, _base) in sources.items()
    }
    candidates["categorical:batter_phand"] = {}
    for label, (frame, _y, _base, source) in blocks.items():
        value = apply_table(frame, tables[source], keys)
        candidates["categorical:batter_phand"][label] = weight * np.where(
            frame["game_type"].eq("R").to_numpy(), value, 0.,
        )

    total = {label: np.zeros(len(frame), dtype=np.float64)
             for label, (frame, _y, _base, _source) in blocks.items()}
    total_gains = {label: 0. for label in blocks}
    remaining = set(candidates)
    selected = []
    scales = (.25, .5, .75, 1., 1.25)
    for step in range(10):
        best = None
        for name in remaining:
            for scale in scales:
                new_gains, marginal = {}, {}
                for label, (_frame, y, base, _source) in blocks.items():
                    correction = total[label] + scale * candidates[name][label]
                    new_gains[label] = score_gain(y, base, correction)
                    marginal[label] = new_gains[label] - total_gains[label]
                if min(marginal.values()) < -1e-9:
                    continue
                rank = (min(new_gains.values()),
                        float(np.mean(list(new_gains.values()))),
                        min(marginal.values()))
                if best is None or rank > best[0]:
                    best = (rank, name, scale, new_gains, marginal)
        if best is None or min(best[4].values()) < .02:
            break
        _rank, name, scale, new_gains, marginal = best
        for label in blocks:
            total[label] += scale * candidates[name][label]
        total_gains = new_gains
        remaining.remove(name)
        selected.append({
            "step": step + 1, "name": name, "scale": scale,
            "marginal_gains": marginal, "total_gains": total_gains.copy(),
        })
        print(json.dumps(selected[-1], indent=2), flush=True)

    report = {"selected": selected, "final_gains": total_gains,
              "candidate_count": len(candidates)}
    output = root / "research/joint_residual_portfolio_v19.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
