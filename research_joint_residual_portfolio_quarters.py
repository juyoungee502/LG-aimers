"""Select residual effects that improve every deployment-like quarter split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_numeric_residual_tables import apply_effect, fit_effect
from research_residual_portfolio_v19 import apply_table, build_table, prepare


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
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
        }
        folds[year]["residual"] = folds[year]["y"] - folds[year]["base"]
    q = {year: np.linspace(0, len(rows[year]), 5, dtype=int) for year in rows}
    sources = {
        "23h1": (rows[2023].iloc[:q[2023][2]],
                  folds[2023]["residual"][:q[2023][2]],
                  folds[2023]["base"][:q[2023][2]]),
        "23": (rows[2023], folds[2023]["residual"], folds[2023]["base"]),
        "24h1": (rows[2024].iloc[:q[2024][2]],
                  folds[2024]["residual"][:q[2024][2]],
                  folds[2024]["base"][:q[2024][2]]),
    }
    blocks = {}
    for index in (2, 3):
        start, stop = q[2023][index], q[2023][index + 1]
        blocks[f"2023_q{index + 1}"] = (
            rows[2023].iloc[start:stop], folds[2023]["y"][start:stop],
            folds[2023]["base"][start:stop], "23h1",
        )
    for index in range(4):
        start, stop = q[2024][index], q[2024][index + 1]
        blocks[f"2023_to_2024_q{index + 1}"] = (
            rows[2024].iloc[start:stop], folds[2024]["y"][start:stop],
            folds[2024]["base"][start:stop], "23",
        )
    for index in (2, 3):
        start, stop = q[2024][index], q[2024][index + 1]
        blocks[f"2024_h1_to_q{index + 1}"] = (
            rows[2024].iloc[start:stop], folds[2024]["y"][start:stop],
            folds[2024]["base"][start:stop], "24h1",
        )

    reports = json.loads(
        (root / "research/numeric_residual_tables_v19.json").read_text("utf-8")
    )
    best = {}
    for row in reports:
        best.setdefault((row["feature"], row["context"]), row)
    configs = [row for row in best.values()
               if row["min_transfer"] >= .03 and abs(row["weight"]) >= .024]
    candidates = {}
    for config in configs:
        name = f"numeric:{config['feature']}:{config['context']}"
        fitted = {
            label: fit_effect(frame, residual, config["feature"], base,
                              config["context"], int(config["n_bins"]),
                              float(config["shrink"]))
            for label, (frame, residual, base) in sources.items()
        }
        candidates[name] = {
            label: float(config["weight"]) * apply_effect(
                frame, config["feature"], base, config["context"], fitted[source],
            )
            for label, (frame, _y, base, source) in blocks.items()
        }
    keys = ["batter_id", "pitcher_hand"]
    cat_tables = {label: build_table(frame, residual, keys, 400.)
                  for label, (frame, residual, _base) in sources.items()}
    candidates["categorical:batter_phand"] = {
        label: .125 * np.where(
            frame["game_type"].eq("R").to_numpy(),
            apply_table(frame, cat_tables[source], keys), 0.,
        ) for label, (frame, _y, _base, source) in blocks.items()
    }

    total = {label: np.zeros(len(frame), dtype=np.float64)
             for label, (frame, _y, _base, _source) in blocks.items()}
    total_gains = {label: 0. for label in blocks}
    remaining, selected = set(candidates), []
    for step in range(12):
        winner = None
        for name in remaining:
            for scale in (.25, .5, .75, 1., 1.25):
                new, marginal = {}, {}
                for label, (_frame, y, base, _source) in blocks.items():
                    new[label] = gain(
                        y, base, total[label] + scale * candidates[name][label]
                    )
                    marginal[label] = new[label] - total_gains[label]
                if min(marginal.values()) < -1e-9:
                    continue
                rank = (min(new.values()), float(np.mean(list(new.values()))),
                        min(marginal.values()))
                if winner is None or rank > winner[0]:
                    winner = (rank, name, scale, new, marginal)
        if winner is None or min(winner[4].values()) < .01:
            break
        _rank, name, scale, new, marginal = winner
        for label in blocks:
            total[label] += scale * candidates[name][label]
        total_gains = new
        remaining.remove(name)
        selected.append({"step": step + 1, "name": name, "scale": scale,
                         "marginal_gains": marginal, "total_gains": new})
        print(json.dumps(selected[-1], indent=2), flush=True)
    result = {"selected": selected, "final_gains": total_gains,
              "candidate_count": len(candidates)}
    output = root / "research/joint_residual_portfolio_quarters_v19.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
