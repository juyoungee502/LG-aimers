#!/usr/bin/env bash
set -Eeuo pipefail

VERSION="v66"
LOG_FILE="training_${VERSION}_research.log"
RESULT_DIR="outputs"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}_research.zip"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" research

required=(
    data/train.csv
    data/trackman_history.csv
    outputs/trackman_pitch_alignment.npz
    outputs/v54_oof_predictions.npz
)
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $path is missing." >&2
        exit 1
    fi
done

{
    for year in 2023 2024; do
        echo "[$(date '+%F %T')] Training paired multitask TabM for ${year}"
        "$PYTHON_BIN" research_v66_multitask_tabm.py \
            --valid-year "$year" --epochs 4 --aux-weight .08
    done
    echo "[$(date '+%F %T')] Auditing fixed cross-year directions"
    "$PYTHON_BIN" research_v66_multitask_audit.py
    echo "[$(date '+%F %T')] v66 research completed"
} 2>&1 | tee "$LOG_FILE"

"$PYTHON_BIN" - "$LOG_FILE" "$RESULT_BUNDLE" <<'PY'
from pathlib import Path
import sys
import zipfile

inputs = [
    Path(sys.argv[1]),
    Path("research/v66_multitask_tabm_2023.npz"),
    Path("research/v66_multitask_tabm_2024.npz"),
    Path("research/v66_multitask_audit.json"),
]
missing = [str(path) for path in inputs if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Missing v66 research artifacts: {missing}")
bundle = Path(sys.argv[2])
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in inputs:
        archive.write(path, path.name)
print(f"Research bundle ready: {bundle}")
PY

echo "Copy this file to your PC: ${RESULT_BUNDLE}"
