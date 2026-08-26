"""Test independent pitch-choice signals on top of the v16 failure prior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import TARGET, bss, reconstruct_labels, signal
from upgrade_v15_to_v16 import history_frame


SPECS = {
    "expected": (500., 30., "expected"),
    "selection": (100., 300., "selection"),
}


def fixed_signal(history, year, spec):
    outcome_k, selection_k, mode = spec
    source = history.loc[history["season"].lt(year)]
    rows = history.loc[history["season"].eq(year)].reset_index(drop=True)
    proxy = history.loc[history["season"].eq(year - 1)].reset_index(drop=True)
    mode_index = 0 if mode == "expected" else 1
    raw = signal(
        source, rows, TARGET, outcome_k, selection_k, "hand_count"
    )[mode_index]
    proxy_raw = signal(
        source, proxy, TARGET, outcome_k, selection_k, "hand_count"
    )[mode_index]
    center = float(proxy_raw[proxy["game_type"].eq("R").to_numpy()].mean())
    result = np.zeros(len(rows), dtype=np.float64)
    regular = rows["game_type"].eq("R").to_numpy()
    result[regular] = raw[regular] - center
    return result, center


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    history = history_frame(data, reconstruct_labels(data))
    with np.load(root / "outputs" / "v16_oof_predictions.npz", allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}

    signals, centers = {}, {}
    for year in (2023, 2024):
        signals[year], centers[year] = {}, {}
        for name, spec in SPECS.items():
            signals[year][name], centers[year][name] = fixed_signal(history, year, spec)

    grids = {
        "expected": np.arange(-.10, .301, .025),
        "selection": np.arange(-.25, 1.251, .05),
    }
    results = []
    for expected_weight in grids["expected"]:
        for selection_weight in grids["selection"]:
            weights = {"expected": expected_weight, "selection": selection_weight}
            full_gains, half_gains = {}, {}
            for year in (2023, 2024):
                mask = oof["season"] == year
                target = oof["target"][mask].astype(float)
                base = oof["blended"][mask].astype(float)
                correction = sum(weights[name] * signals[year][name] for name in SPECS)
                prediction = np.clip(base + correction, .005, .995)
                full_gains[str(year)] = bss(target, prediction) - bss(target, base)
                halfway = len(target) // 2
                half_gains[str(year)] = [
                    bss(target[:halfway], prediction[:halfway])
                    - bss(target[:halfway], base[:halfway]),
                    bss(target[halfway:], prediction[halfway:])
                    - bss(target[halfway:], base[halfway:]),
                ]
            result = {
                "expected_weight": float(expected_weight),
                "selection_weight": float(selection_weight),
                "gain_2023": full_gains["2023"], "gain_2024": full_gains["2024"],
                "min_year": min(full_gains.values()),
                "mean_year": float(np.mean(list(full_gains.values()))),
                "half_gains": half_gains,
                "min_half": min(value for values in half_gains.values() for value in values),
            }
            results.append(result)
    results.sort(
        key=lambda item: (item["min_year"], item["min_half"], item["mean_year"]),
        reverse=True,
    )
    report = {
        "base": "v16 fixed pitch-failure prior",
        "fixed_proxy_centers": centers,
        "top": results[:50],
    }
    output = root / "research" / "pitch_prior_combinations.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
