"""Screen direct fine-pitch command priors on top of v23."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss, reconstruct_labels
from research_trackman_failure_prior import TARGET, aligned_history, selection_delta


def segment_curve(target, base, signal, rows):
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
    }
    curves = {}
    residual = target - base
    for name, mask in masks.items():
        if not mask.any():
            continue
        reference = float(target[mask].mean() * (1.0 - target[mask].mean()))
        linear = 200000.0 * float(np.mean(signal[mask] * residual[mask])) / reference
        quadratic = 100000.0 * float(np.mean(signal[mask] ** 2)) / reference
        curves[name] = (linear, quadratic)
    return curves


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False,
    )
    labels = reconstruct_labels(data)
    history = aligned_history(root, data, labels)
    with np.load(root / "outputs" / "v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}

    configurations = [
        (outcome_k, selection_k)
        for outcome_k in (10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
        for selection_k in (30.0, 100.0, 300.0, 600.0)
    ]
    prepared = {}
    for year in (2023, 2024):
        source = history.loc[history["season"].lt(year)]
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
        regular = rows["game_type"].eq("R").to_numpy()
        proxy_regular = proxy["game_type"].eq("R").to_numpy()
        for outcome_k, selection_k in configurations:
            raw = selection_delta(
                source, rows, TARGET, outcome_k, selection_k,
            )
            proxy_raw = selection_delta(
                source, proxy, TARGET, outcome_k, selection_k,
            )
            center = float(proxy_raw[proxy_regular].mean())
            signal = np.zeros(len(rows), dtype=float)
            signal[regular] = raw[regular] - center
            prepared[(year, outcome_k, selection_k)] = (signal, center)
        print(f"Prepared {year}: history={len(source)}", flush=True)

    reports = []
    for outcome_k, selection_k in configurations:
        curves = {}
        centers = {}
        for year in (2023, 2024):
            mask = oof["season"] == year
            y = oof["target"][mask].astype(float)
            base = oof["blended"][mask].astype(float)
            rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
            signal, center = prepared[(year, outcome_k, selection_k)]
            curves[str(year)] = segment_curve(y, base, signal, rows)
            centers[str(year)] = center
        for weight in np.arange(-2.0, 2.001, .05):
            year_gains = {}
            all_segments = []
            for year in ("2023", "2024"):
                gains = {
                    name: linear * weight - quadratic * weight**2
                    for name, (linear, quadratic) in curves[year].items()
                }
                year_gains[year] = gains
                all_segments.extend(gains.values())
            reports.append({
                "outcome_k": outcome_k, "selection_k": selection_k,
                "weight": float(weight), "gains": year_gains,
                "centers": centers,
                "min_year": min(
                    year_gains["2023"]["all"], year_gains["2024"]["all"],
                ),
                "min_half": min(
                    year_gains[year][half]
                    for year in ("2023", "2024")
                    for half in ("first_half", "second_half")
                ),
                "min_segment": min(all_segments),
            })
    reports.sort(
        key=lambda row: (
            row["min_year"], row["min_half"], row["gains"]["2024"]["all"],
        ), reverse=True,
    )
    result = {
        "aligned_rows": len(history), "target": TARGET,
        "official_evaluation_data_used": False,
        "current_validation_pitch_type_used": False,
        "reports": reports,
    }
    output = root / "research" / "v23_fine_pitch_success.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"top": reports[:60]}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
