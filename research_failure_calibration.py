"""Find leakage-safe calibration and blend weights for failure specialists.

Every calibration target is frozen from seasons strictly before the validation
season.  The script never uses the validation labels to set an offset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import TARGET_COL
from research_inferred_pitch_priors import bss, reconstruct_labels


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def shift_mean(probability, target_mean):
    z = logit(probability)
    offset = 0.0
    for _ in range(50):
        shifted = sigmoid(z + offset)
        error = float(shifted.mean()) - float(target_mean)
        if abs(error) < 1e-13:
            break
        offset -= error / max(float(np.mean(shifted * (1.0 - shifted))), 1e-10)
    return sigmoid(z + offset), float(offset)


def historical_rate(data, values, valid_year, mode):
    if mode == "previous":
        mask = data["season"].eq(valid_year - 1).to_numpy()
        return float(values[mask].mean())
    if mode == "previous2":
        mask = data["season"].isin((valid_year - 2, valid_year - 1)).to_numpy()
        return float(values[mask].mean())
    if mode == "weighted3":
        mask = data["season"].lt(valid_year).to_numpy()
        age = (valid_year - 1) - data.loc[mask, "season"].to_numpy(float)
        weight = np.exp(-np.log(2.0) * age / 3.0)
        return float(np.average(values[mask], weights=weight))
    raise ValueError(mode)


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    usable = labels[["reverse", "middle", "wayoff"]].notna().all(axis=1).to_numpy()
    target = data[TARGET_COL].to_numpy(float)
    middle_success = 1.0 - labels["middle"].fillna(0).to_numpy(float)
    reports = []
    cache = {}
    for year in (2023, 2024):
        with np.load(root / "research" / f"failure_specialists_{year}.npz") as loaded:
            q = {key: loaded[key] for key in loaded.files}
        for variant_index, variant in enumerate(q["variants"]):
            raw_all = q["predictions"][variant_index, :, 3].astype(float)
            raw_middle = q["predictions"][variant_index, :, 4].astype(float)
            cache[(year, str(variant))] = (q, raw_all, raw_middle)

    modes = ("raw", "previous", "previous2", "weighted3")
    for variant in ("weighted_depth6", "uniform_depth8"):
        for all_mode in modes:
            for middle_mode in modes:
                fold_predictions = {}
                offsets = {}
                for year in (2023, 2024):
                    q, raw_all, raw_middle = cache[(year, variant)]
                    if all_mode == "raw":
                        p_all, all_offset = raw_all, 0.0
                    else:
                        center = historical_rate(data, target, year, all_mode)
                        p_all, all_offset = shift_mean(raw_all, center)
                    if middle_mode == "raw":
                        p_middle, middle_offset = raw_middle, 0.0
                    else:
                        center = historical_rate(
                            data.loc[usable].reset_index(drop=True),
                            middle_success[usable], year, middle_mode,
                        )
                        p_middle, middle_offset = shift_mean(raw_middle, center)
                    fold_predictions[year] = (q, p_all, p_middle)
                    offsets[year] = (all_offset, middle_offset)
                for weight_all in np.arange(0.0, 0.151, 0.01):
                    for weight_middle in np.arange(0.0, 0.021, 0.005):
                        gains = {}
                        all_slices = []
                        for year in (2023, 2024):
                            q, p_all, p_middle = fold_predictions[year]
                            y = q["target"].astype(float)
                            base = q["base"].astype(float)
                            prediction = sigmoid(
                                (1.0 - weight_all - weight_middle) * logit(base)
                                + weight_all * logit(p_all)
                                + weight_middle * logit(p_middle)
                            )
                            half = len(y) // 2
                            slices = ((y, prediction, base),
                                      (y[:half], prediction[:half], base[:half]),
                                      (y[half:], prediction[half:], base[half:]))
                            values = [bss(a, b) - bss(a, c) for a, b, c in slices]
                            gains[str(year)] = values
                            all_slices.extend(values[1:])
                        reports.append({
                            "variant": variant, "all_mode": all_mode,
                            "middle_mode": middle_mode,
                            "weight_all": float(weight_all),
                            "weight_middle": float(weight_middle),
                            "gain_2023": gains["2023"][0],
                            "gain_2024": gains["2024"][0],
                            "min_year": min(gains["2023"][0], gains["2024"][0]),
                            "min_half_all_years": min(all_slices),
                            "offsets": {str(k): list(v) for k, v in offsets.items()},
                        })
    reports.sort(
        key=lambda row: (row["min_year"], row["min_half_all_years"],
                         row["gain_2024"]), reverse=True,
    )
    output = root / "research" / "failure_calibration_report.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
