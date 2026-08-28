#!/usr/bin/env bash
set -Eeuo pipefail

VERSION="v60"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
AUDIT_FILE="research/v59_group_stability.json"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" research

if [[ ! -f outputs/v54_oof_predictions.npz ]]; then
    echo "v54 OOF is missing; building the v54 base first."
    bash run_v54.sh
fi

required=(
    data/train.csv
    outputs/v54_oof_predictions.npz
    submit/model/metadata.json
)
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $path is missing." >&2
        exit 1
    fi
done

{
    for year in 2023 2024; do
        if [[ ! -f "research/v59_f_fraction_s3_${year}.npz" ]]; then
            "$PYTHON_BIN" research_v59_f_fraction_specialist.py \
                --valid-year "$year" --n-seeds 3 --seed-offset 0 \
                --task-type GPU --devices 0
        fi
        if [[ ! -f "research/v59_f_fraction_s3_o3_${year}.npz" ]]; then
            "$PYTHON_BIN" research_v59_f_fraction_specialist.py \
                --valid-year "$year" --n-seeds 3 --seed-offset 3 \
                --task-type GPU --devices 0
        fi
    done
    echo "[$(date '+%F %T')] Auditing independent seed groups"
    "$PYTHON_BIN" research_v59_group_stability.py
    echo "[$(date '+%F %T')] Training fraction-confidence ensemble for ${VERSION}"
    "$PYTHON_BIN" train_v60_fraction_confidence.py --task-type GPU --devices 0
    "$PYTHON_BIN" build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v60_fraction_confidence

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q "$SUBMISSION_FILE" -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
from recent_window_features import recent_window_features

data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
regular = data.loc[data["game_type"].eq("R")].head(3).copy()
futures = data.loc[data["game_type"].eq("F")].copy()
recent = recent_window_features(futures)
futures = futures.loc[recent["recent1_reduced_n"].ge(30.)].tail(2).copy()
if len(futures) != 2:
    raise RuntimeError("Could not create v60 selected-row smoke sample")
# Frozen inference history ends at the training boundary.  Move the synthetic
# rows into a later-season exposure state so the v60 gate is exercised.
futures["asof_pitcher_n"] = futures["asof_pitcher_n"] + 1000.
sample = pd.concat([regular, futures], ignore_index=True).drop(
    columns=["control_success"],
)
sample.to_csv(sys.argv[2], index=False, encoding="utf-8")
PY
    (
        cd "$smoke_dir"
        "$PYTHON_BIN" script.py
    )
    "$PYTHON_BIN" - "$smoke_dir/output/submission.csv" <<'PY'
import sys
import pandas as pd

result = pd.read_csv(sys.argv[1])
if len(result) != 5 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v60 smoke test failed")
print(f"Packaged v60 smoke test passed: rows={len(result)}")
PY
    echo "[$(date '+%F %T')] GPU training, validation, build, and smoke test completed"
} 2>&1 | tee "$LOG_FILE"

"$PYTHON_BIN" - \
    "$SUBMISSION_FILE" "$LOG_FILE" "$OOF_FILE" "$AUDIT_FILE" \
    "$RESULT_BUNDLE" <<'PY'
from pathlib import Path
import sys
import zipfile

inputs = [Path(value) for value in sys.argv[1:5]]
bundle = Path(sys.argv[5])
missing = [str(path) for path in inputs if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Missing result artifacts: {missing}")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in inputs:
        archive.write(path, path.name)
print(f"Result bundle ready: {bundle}")
PY

echo "Copy this file to your PC: ${RESULT_BUNDLE}"
