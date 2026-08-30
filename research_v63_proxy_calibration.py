"""Measure the deployed v62 probability level on a 2025-shaped proxy.

The proxy changes only the season value of official 2024 training rows.  It is
used to audit the packaged inference path and never contributes labels,
aggregates, or predictions at evaluation time.
"""
from __future__ import annotations

import io
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
REFERENCE_HIDDEN_RATE = 0.47827


def extract_submission(result_path: Path, stage: Path) -> None:
    with zipfile.ZipFile(result_path) as outer:
        payload = outer.read("submission_v62.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        inner.extractall(stage)


def build_proxy(output: Path) -> dict:
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
    return {
        "rows": int(rows),
        "source_2024_target_mean": float(target_sum / rows),
    }


def main() -> None:
    result_path = ROOT / "outputs/results_v62.zip"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with tempfile.TemporaryDirectory(prefix="v63_proxy_") as temporary:
        stage = Path(temporary)
        extract_submission(result_path, stage)
        source = build_proxy(stage / "data/test.csv")
        subprocess.run([sys.executable, "script.py"], cwd=stage, check=True)
        prediction = pd.read_csv(stage / "output/submission.csv")[TARGET].to_numpy(float)
    denominator = REFERENCE_HIDDEN_RATE * (1.0 - REFERENCE_HIDDEN_RATE)
    offset = float(REFERENCE_HIDDEN_RATE - prediction.mean())
    report = {
        "version": "v62_public_residual_frontier",
        "proxy": source,
        "prediction": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "quantiles": {
                str(q): float(np.quantile(prediction, q))
                for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
        },
        "external_hidden_rate_reference": REFERENCE_HIDDEN_RATE,
        "reference_probability_offset": offset,
        "reference_maximum_bss_gain": float(100_000.0 * offset**2 / denominator),
        "proxy_only": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    output = ROOT / "research/v63_proxy_calibration.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
