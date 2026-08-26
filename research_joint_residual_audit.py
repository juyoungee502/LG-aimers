"""Audit the selected residual portfolio on finer, previously unused time splits."""
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

    numeric_rows = json.loads(
        (root / "research/numeric_residual_tables_v19.json").read_text("utf-8")
    )
    best_numeric = {}
    for row in numeric_rows:
        best_numeric.setdefault((row["feature"], row["context"]), row)
    selection = json.loads(
        (root / "research/joint_residual_portfolio_v19.json").read_text("utf-8")
    )["selected"]

    q = {year: np.linspace(0, len(rows[year]), 5, dtype=int) for year in rows}
    transfers = []
    # Source year -> each target-year quarter is closest to 2024 -> 2025 use.
    for target_q in range(4):
        transfers.append((f"2023_to_2024_q{target_q + 1}", 2023, 0, len(rows[2023]),
                          2024, q[2024][target_q], q[2024][target_q + 1]))
    # Within-year tests stress recency and smaller source samples.
    for target_q in (2, 3):
        transfers.append((f"2023_h1_to_q{target_q + 1}", 2023, 0, q[2023][2],
                          2023, q[2023][target_q], q[2023][target_q + 1]))
        transfers.append((f"2024_h1_to_q{target_q + 1}", 2024, 0, q[2024][2],
                          2024, q[2024][target_q], q[2024][target_q + 1]))
    for year in (2023, 2024):
        transfers.append((f"{year}_q1_to_q2", year, 0, q[year][1],
                          year, q[year][1], q[year][2]))

    reports = []
    for label, sy, ss, se, ty, ts, te in transfers:
        source = rows[sy].iloc[ss:se]
        source_residual = folds[sy]["residual"][ss:se]
        source_base = folds[sy]["base"][ss:se]
        target = rows[ty].iloc[ts:te]
        y = folds[ty]["y"][ts:te]
        base = folds[ty]["base"][ts:te]
        total = np.zeros(len(target), dtype=np.float64)
        prefix = []
        previous = 0.
        for chosen in selection:
            name, scale = chosen["name"], float(chosen["scale"])
            if name == "categorical:batter_phand":
                keys = ["batter_id", "pitcher_hand"]
                table = build_table(source, source_residual, keys, 400.)
                value = .125 * apply_table(target, table, keys)
                value = np.where(target["game_type"].eq("R").to_numpy(), value, 0.)
            else:
                _, feature, context = name.split(":", 2)
                config = best_numeric[(feature, context)]
                fitted = fit_effect(
                    source, source_residual, feature, source_base, context,
                    int(config["n_bins"]), float(config["shrink"]),
                )
                value = float(config["weight"]) * apply_effect(
                    target, feature, base, context, fitted,
                )
            total += scale * value
            current = gain(y, base, total)
            prefix.append({"n": len(prefix) + 1, "name": name,
                           "gain": current, "marginal": current - previous})
            previous = current
        reports.append({"transfer": label, "source_n": len(source),
                        "target_n": len(target), "prefix": prefix})

    summary = []
    for n in range(1, len(selection) + 1):
        values = [report["prefix"][n - 1]["gain"] for report in reports]
        summary.append({"n": n, "min": min(values), "mean": float(np.mean(values)),
                        "positive": int(np.sum(np.asarray(values) > 0.)),
                        "total": len(values), "gains": values})
    result = {"summary": summary, "transfers": reports}
    output = root / "research/joint_residual_audit_v19.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
