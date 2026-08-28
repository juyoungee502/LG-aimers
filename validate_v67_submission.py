"""Run a packaged submission end-to-end on a small schema-valid batch."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="v67_submission_") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(directory)
        (directory / "data").mkdir()
        (directory / "output").mkdir()
        train = pd.read_csv(
            root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
        )
        test = train.loc[train["season"].eq(2024)].head(args.rows).drop(
            columns=["control_success"]
        )
        test.to_csv(directory / "data/test.csv", index=False, encoding="utf-8")
        subprocess.run([sys.executable, "script.py"], cwd=directory, check=True)
        output = pd.read_csv(directory / "output/submission.csv")
        if (
            len(output) != args.rows
            or not output["row_id"].is_unique
            or not output["control_success"].between(0., 1.).all()
        ):
            raise ValueError("Invalid smoke-test submission")
        print(output["control_success"].describe().to_string())
        print("FULL_ZIP_SMOKE_OK")


if __name__ == "__main__":
    main()
