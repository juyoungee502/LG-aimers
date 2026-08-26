"""Validate trained artifacts and build the code-submission ZIP."""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path

REQUIRED_MODELS = (
    "metadata.json", "lgb_0.txt", "lgb_1.txt",
    "catboost_0.cbm", "catboost_1.cbm", "catboost_2.cbm",
    "catboost_other_0.cbm", "catboost_other_1.cbm", "catboost_other_2.cbm",
    "catboost_two_strike_0.cbm", "catboost_two_strike_1.cbm", "catboost_two_strike_2.cbm",
    "catboost_categorical_0.cbm", "catboost_categorical_1.cbm", "catboost_categorical_2.cbm",
    "catboost_categorical_other_0.cbm", "catboost_categorical_other_1.cbm", "catboost_categorical_other_2.cbm",
    "catboost_categorical_two_strike_0.cbm", "catboost_categorical_two_strike_1.cbm", "catboost_categorical_two_strike_2.cbm",
    "catboost_brier_0.cbm", "catboost_brier_1.cbm", "catboost_brier_2.cbm",
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submit-dir", default="submit")
    p.add_argument("--output", default="submission_v3.zip")
    p.add_argument("--expected-version", default="v16_pitch_failure_prior")
    args = p.parse_args(); root = Path(args.submit_dir)
    model_names = list(REQUIRED_MODELS)
    if args.expected_version in (
        "v14_weighted_catboost", "v15_weighted_categorical_specialist",
        "v16_pitch_failure_prior",
    ):
        model_names.extend(f"catboost_weighted_{index}.cbm" for index in range(3))
    if args.expected_version in (
        "v15_weighted_categorical_specialist", "v16_pitch_failure_prior",
    ):
        model_names.extend(
            f"catboost_weighted_categorical_{label}_{index}.cbm"
            for label in ("other", "two_strike") for index in range(3)
        )
    required = [
        root / "script.py", root / "requirements.txt", Path("feature_engineering.py"),
        Path("residual_effects.py"),
    ] + [root / "model" / n for n in model_names]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing submission files: {missing}")
    metadata = json.loads((root / "model" / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != args.expected_version:
        raise ValueError(f"Unexpected model version: {metadata.get('version')}")
    output = Path(args.output)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in required:
            arcname = path.name if path in (Path("feature_engineering.py"), Path("residual_effects.py")) else path.relative_to(root).as_posix()
            archive.write(path, arcname)
    print(f"Created {output.resolve()} ({output.stat().st_size / 1024**2:.2f} MiB)")
    with zipfile.ZipFile(output) as archive:
        print("Contents:", archive.namelist())

if __name__ == "__main__": main()
