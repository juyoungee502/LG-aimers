#!/usr/bin/env bash
set -Eeuo pipefail

VERSION="v38"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" research

if [[ ! -f outputs/v24_oof_predictions.npz ]]; then
    echo "v24 OOF is missing; building the v24 base first."
    bash run_v24.sh
fi

required=(
    data/train.csv
    outputs/v23_oof_predictions.npz
    outputs/v24_oof_predictions.npz
    submit/model/metadata.json
)
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $path is missing." >&2
        exit 1
    fi
done

{
    if [[ ! -f research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz ]]; then
        echo "[$(date '+%F %T')] Building v34 chronological GPU audit"
        "$PYTHON_BIN" research_v34_categorical_failure_fix.py \
            --valid-year 2024 --profile lowcard_no_ids --half-life 2 \
            --task-type GPU --devices 0
    fi
    if [[ ! -f research/v35_lowcard_direct_hl2_s3_2024.npz ]]; then
        echo "[$(date '+%F %T')] Building v35 chronological GPU audit"
        "$PYTHON_BIN" research_v35_lowcard_direct_cat.py \
            --valid-year 2024 --n-seeds 3 --half-life 2 \
            --task-type GPU --devices 0
    fi
    echo "[$(date '+%F %T')] Training low-cardinality ensemble for ${VERSION}"
    "$PYTHON_BIN" train_v38_lowcard_ensemble.py --task-type GPU --devices 0
    "$PYTHON_BIN" build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v38_lowcard_ensemble
    echo "[$(date '+%F %T')] GPU training, validation, and build completed"
} 2>&1 | tee "$LOG_FILE"

"$PYTHON_BIN" - "$SUBMISSION_FILE" "$LOG_FILE" "$OOF_FILE" "$RESULT_BUNDLE" <<'PY'
from pathlib import Path
import sys
import zipfile

inputs = [Path(value) for value in sys.argv[1:4]]
bundle = Path(sys.argv[4])
missing = [str(path) for path in inputs if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Missing result artifacts: {missing}")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in inputs:
        archive.write(path, path.name)
print(f"Result bundle ready: {bundle}")
PY

echo "Copy this file to your PC: ${RESULT_BUNDLE}"
