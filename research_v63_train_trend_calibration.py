"""Select a conservative train-only probability calibration for v63."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
OFFSET = -0.0015
V61_PUBLIC = 1132.7
V62_PUBLIC = 1132.9
CLIP = (0.005, 0.995)


def constant_gain(prediction_mean: float, target_rate: float, offset: float) -> float:
    denominator = target_rate * (1.0 - target_rate)
    bias = prediction_mean - target_rate
    return float(-100_000.0 * (2.0 * offset * bias + offset**2) / denominator)


def main() -> None:
    proxy = json.loads(
        (ROOT / "research/v63_proxy_calibration_v61.json").read_text(encoding="utf-8")
    )
    if proxy["version"] != "v61" or not proxy["proxy_only"]:
        raise ValueError("A train-only v61 proxy audit is required")
    annual = {
        int(year): float(rate)
        for year, rate in proxy["train_only_rate_forecast"]["annual_rates"].items()
    }
    prediction_mean = float(proxy["prediction"]["mean"])
    trend_rate = float(proxy["train_only_rate_forecast"]["forecast_2025"])
    years = np.asarray(sorted(annual), dtype=float)
    rates = np.asarray([annual[int(year)] for year in years], dtype=float)
    slope_all, intercept_all = np.polyfit(years, rates, 1)
    all_year_rate = float(intercept_all + slope_all * 2025.0)
    persistence_rate = annual[2024]

    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    active = season == 2024
    candidate = np.clip(base[active] + OFFSET, *CLIP)
    local_gain = float(
        bss(target[active], candidate) - bss(target[active], base[active])
    )
    scenarios = {
        "2024_rate_persistence": {
            "target_rate": persistence_rate,
            "gain": constant_gain(prediction_mean, persistence_rate, OFFSET),
        },
        "official_2020_2024_trend": {
            "target_rate": trend_rate,
            "gain": constant_gain(prediction_mean, trend_rate, OFFSET),
        },
        "official_2019_2024_trend": {
            "target_rate": all_year_rate,
            "gain": constant_gain(prediction_mean, all_year_rate, OFFSET),
        },
    }
    central_gain = scenarios["official_2020_2024_trend"]["gain"]
    report = {
        "baseline": "v61_public_complete_shape",
        "public_feedback": {
            "v61": V61_PUBLIC,
            "v62": V62_PUBLIC,
            "v62_gain": V62_PUBLIC - V61_PUBLIC,
            "decision": "discard all v62 structural corrections",
        },
        "configuration": {
            "probability_offset": OFFSET,
            "full_train_trend_offset": float(trend_rate - prediction_mean),
            "fraction_of_full_offset": float(OFFSET / (trend_rate - prediction_mean)),
            "proxy_prediction_mean": prediction_mean,
            "train_only_2025_rate_forecast": trend_rate,
        },
        "strict_oof_2024_gain": local_gain,
        "train_only_scenarios": scenarios,
        "projection": {
            "gain": central_gain,
            "score": float(V61_PUBLIC + central_gain),
            "range": [1131.0, 1145.0],
        },
        "proxy_uses_training_rows_only": True,
        "leaderboard_inferred_target_rate_used": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    output = ROOT / "research/v63_train_trend_calibration.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
