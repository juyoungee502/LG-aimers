"""Measure the deployed v62 probability level on a 2025-shaped proxy.

The proxy changes only the season value of official 2024 training rows.  It is
used to audit the packaged inference path and never contributes labels,
aggregates, or predictions at evaluation time.
"""
from __future__ import annotations

import io
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"


def extract_submission(result_path: Path, stage: Path, version: int) -> None:
    with zipfile.ZipFile(result_path) as outer:
        payload = outer.read(f"submission_v{version}.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        inner.extractall(stage)


def build_proxy(output: Path) -> tuple[dict, pd.DataFrame]:
    parts = []
    target_sum = 0.0
    rows = 0
    for chunk in pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig",
        low_memory=False, chunksize=100_000,
    ):
        selected = chunk.loc[chunk["season"].eq(2024)].copy()
        if selected.empty:
            continue
        target_sum += float(selected[TARGET].sum())
        rows += len(selected)
        selected["season"] = 2025
        parts.append(selected.drop(columns=[TARGET]))
    proxy = pd.concat(parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    proxy.to_csv(output, index=False, encoding="utf-8")
    annual = pd.concat([
        chunk.groupby("season")[TARGET].agg(["sum", "count"])
        for chunk in pd.read_csv(
            ROOT / "data/train.csv", usecols=["season", TARGET],
            encoding="utf-8-sig", chunksize=200_000,
        )
    ]).groupby(level=0).sum()
    annual["rate"] = annual["sum"] / annual["count"]
    return {
        "rows": int(rows),
        "source_2024_target_mean": float(target_sum / rows),
    }, annual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(61, 62), default=62)
    args = parser.parse_args()
    result_path = ROOT / f"outputs/results_v{args.version}.zip"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with tempfile.TemporaryDirectory(prefix="v63_proxy_") as temporary:
        stage = Path(temporary)
        extract_submission(result_path, stage, args.version)
        source, annual = build_proxy(stage / "data/test.csv")
        subprocess.run([sys.executable, "script.py"], cwd=stage, check=True)
        prediction = pd.read_csv(stage / "output/submission.csv")[TARGET].to_numpy(float)
    trend_years = annual.loc[2020:2024].index.to_numpy(float)
    trend_rates = annual.loc[2020:2024, "rate"].to_numpy(float)
    slope, intercept = np.polyfit(trend_years, trend_rates, 1)
    train_only_rate = float(intercept + slope * 2025.0)
    denominator = train_only_rate * (1.0 - train_only_rate)
    offset = float(train_only_rate - prediction.mean())
    report = {
        "version": f"v{args.version}",
        "proxy": source,
        "prediction": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "quantiles": {
                str(q): float(np.quantile(prediction, q))
                for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
        },
        "train_only_rate_forecast": {
            "method": "OLS trend over official 2020-2024 annual target rates",
            "annual_rates": {
                str(int(year)): float(rate)
                for year, rate in annual["rate"].items()
            },
            "slope": float(slope),
            "forecast_2025": train_only_rate,
        },
        "train_only_probability_offset": offset,
        "train_only_maximum_bss_gain": float(100_000.0 * offset**2 / denominator),
        "proxy_only": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    output = ROOT / f"research/v63_proxy_calibration_v{args.version}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
