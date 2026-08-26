"""Re-evaluate stored failure specialists with R/F regime gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", usecols=["season", "game_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    cache = {}
    for feature_set in ("current", "prior_context"):
        suffix = "" if feature_set == "current" else "_prior_context"
        for year in (2023, 2024):
            with np.load(root / f"research/failure_specialists_{year}{suffix}.npz") as loaded:
                q = {key: loaded[key] for key in loaded.files}
            index = list(q["variants"].astype(str)).index("uniform_depth8")
            cache[(feature_set, year)] = (
                q["target"].astype(float), q["base"].astype(float),
                q["predictions"][index, :, 3].astype(float),
                q["predictions"][index, :, 4].astype(float),
                data.loc[data["season"].eq(year), "game_type"].eq("R").to_numpy(),
            )

    reports = []
    for feature_set in ("current", "prior_context"):
        for gate in ("R", "F", "all"):
            for weight_all in np.arange(0., .201, .005):
                for weight_middle in np.arange(0., .031, .005):
                    gains, half_gains = {}, []
                    for year in (2023, 2024):
                        y, base, p_all, p_middle, regular = cache[(feature_set, year)]
                        active = np.ones(len(y), dtype=bool)
                        if gate == "R":
                            active = regular
                        elif gate == "F":
                            active = ~regular
                        prediction = base.copy()
                        prediction[active] = sigmoid(
                            (1. - weight_all - weight_middle) * logit(base[active])
                            + weight_all * logit(p_all[active])
                            + weight_middle * logit(p_middle[active])
                        )
                        midpoint = len(y) // 2
                        values = [
                            bss(y, prediction) - bss(y, base),
                            bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
                            bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
                        ]
                        gains[str(year)] = values
                        half_gains.extend(values[1:])
                    reports.append({
                        "feature_set": feature_set, "gate": gate,
                        "weight_all": float(weight_all),
                        "weight_middle": float(weight_middle),
                        "gain_2023": gains["2023"][0], "gain_2024": gains["2024"][0],
                        "min_year": min(gains["2023"][0], gains["2024"][0]),
                        "min_half": min(half_gains),
                    })
    reports.sort(
        key=lambda row: (row["min_year"], row["min_half"], row["gain_2024"]),
        reverse=True,
    )
    output = root / "research/failure_gating_report.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:80], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
