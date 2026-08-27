"""Audit fine-pitch failure-selection axes on top of v23.

The fine pitch label and detailed failure labels are reconstructed only for
historical training rows.  Each validation direction is a frozen lookup made
from earlier seasons and is centered on the prior-season training proxy.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss, reconstruct_labels
from research_trackman_failure_prior import (
    FAILURES, aligned_history, selection_delta,
)


CONFIGS = tuple(
    itertools.product((5.0, 10.0, 20.0, 50.0), (100.0, 300.0, 600.0))
)
WEIGHT_VALUES = np.arange(-1.5, 2.001, .25)


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }


def curve(target, base, axes, mask):
    uncertainty = float(target[mask].mean() * (1.0 - target[mask].mean()))
    residual = target[mask] - base[mask]
    selected = axes[mask]
    return {
        "linear": (
            200000.0 * np.mean(selected * residual[:, None], axis=0) / uncertainty
        ).tolist(),
        "quadratic": (
            100000.0 * (selected.T @ selected) / (mask.sum() * uncertainty)
        ).tolist(),
    }


def approximate(values, weights):
    linear = np.asarray(values["linear"])
    quadratic = np.asarray(values["quadratic"])
    weights = np.asarray(weights)
    return float(linear @ weights - weights @ quadratic @ weights)


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    labels = reconstruct_labels(data)
    history = aligned_history(root, data, labels)
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        oof = {key: source[key] for key in source.files}

    years = {}
    for year in (2023, 2024):
        fold = oof["season"] == year
        target = oof["target"][fold].astype(float)
        base = oof["blended"][fold].astype(float)
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        regular = rows["game_type"].eq("R").to_numpy()
        if not np.allclose(target, rows["control_success"]):
            raise ValueError(f"v23 rows do not align for {year}")
        source_history = history.loc[history["season"].lt(year)]
        proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
        proxy_regular = proxy["game_type"].eq("R").to_numpy()
        config_axes = {}
        config_centers = {}
        for outcome_k, selection_k in CONFIGS:
            members, centers = [], []
            for failure in FAILURES:
                raw = selection_delta(
                    source_history, rows, failure, outcome_k, selection_k,
                )
                proxy_raw = selection_delta(
                    source_history, proxy, failure, outcome_k, selection_k,
                )
                center = float(proxy_raw[proxy_regular].mean())
                direction = np.zeros(len(rows), dtype=np.float64)
                direction[regular] = raw[regular] - center
                members.append(direction)
                centers.append(center)
            config_axes[(outcome_k, selection_k)] = np.column_stack(members)
            config_centers[(outcome_k, selection_k)] = centers
        years[year] = {
            "target": target, "base": base, "rows": rows,
            "masks": masks(rows), "axes": config_axes,
            "centers": config_centers,
        }
        print(f"Prepared fine failure axes for {year}", flush=True)

    approximate_reports = []
    for outcome_k, selection_k in CONFIGS:
        curves = {
            str(year): {
                name: curve(
                    item["target"], item["base"],
                    item["axes"][(outcome_k, selection_k)], mask,
                )
                for name, mask in item["masks"].items() if mask.any()
            }
            for year, item in years.items()
        }
        for weights in itertools.product(WEIGHT_VALUES, repeat=3):
            if not any(abs(value) > 1e-12 for value in weights):
                continue
            gains = {
                year: {
                    name: approximate(values, weights)
                    for name, values in year_curves.items()
                }
                for year, year_curves in curves.items()
            }
            temporal = [
                value for year_gains in gains.values()
                for name, value in year_gains.items() if name != "all"
            ]
            approximate_reports.append({
                "outcome_k": outcome_k, "selection_k": selection_k,
                "weights": list(weights), "gains": gains,
                "min_segment": min(temporal),
                "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                "mean_year": np.mean([
                    gains["2023"]["all"], gains["2024"]["all"],
                ]),
            })

    chosen = {}
    ranking_keys = (
        lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
        lambda row: (row["min_year"], row["min_segment"], row["mean_year"]),
        lambda row: (row["mean_year"], row["min_year"], row["min_segment"]),
    )
    for ranking in ranking_keys:
        for row in sorted(approximate_reports, key=ranking, reverse=True)[:250]:
            key = (
                row["outcome_k"], row["selection_k"], *row["weights"],
            )
            chosen[key] = row

    exact_reports = []
    for row in chosen.values():
        gains = {}
        temporal = []
        for year, item in years.items():
            axes = item["axes"][(row["outcome_k"], row["selection_k"])]
            candidate = np.clip(
                item["base"] + axes @ np.asarray(row["weights"]), .005, .995,
            )
            gains[str(year)] = {
                name: bss(item["target"][mask], candidate[mask])
                - bss(item["target"][mask], item["base"][mask])
                for name, mask in item["masks"].items() if mask.any()
            }
            temporal.extend(
                value for name, value in gains[str(year)].items() if name != "all"
            )
        exact_reports.append({
            "outcome_k": row["outcome_k"],
            "selection_k": row["selection_k"],
            "weights": row["weights"], "gains": gains,
            "min_segment": min(temporal),
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "mean_year": np.mean([
                gains["2023"]["all"], gains["2024"]["all"],
            ]),
            "centers": {
                str(year): years[year]["centers"][
                    (row["outcome_k"], row["selection_k"])
                ] for year in years
            },
        })

    names = ("maximin_segment", "maximin_year", "best_mean")
    rankings = {
        name: sorted(exact_reports, key=ranking, reverse=True)[:100]
        for name, ranking in zip(names, ranking_keys)
    }
    output = root / "research/v23_fine_pitch_failure.json"
    output.write_text(json.dumps({
        "labels": list(FAILURES), "aligned_rows": len(history),
        "rankings": rankings,
    }, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:15] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
