#!/usr/bin/env bash
set -Eeuo pipefail
VERSION="v24"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" research

required=(
    data/train.csv
    data/trackman_history.csv
    outputs/trackman_pitch_alignment.npz
    outputs/v23_oof_predictions.npz
    submit/model/metadata.json
)
for year in 2023 2024; do
    required+=(
        "research/v23_trackman_no_month_${year}.npz"
        "research/v23_prior_command_context_${year}.npz"
        "research/v23_prior_command_context_${year}_w1.npz"
        "research/v23_conditional_resolution_${year}.npz"
    )
done
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $path is missing." >&2
        exit 1
    fi
done

{
    echo "[$(date '+%F %T')] Training robust command/resolution candidate for ${VERSION}"
    python train_v24_robust_candidate.py --task-type GPU --devices 0
    python build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v24_robust_command_resolution
    echo "[$(date '+%F %T')] Training and submission build completed"
} 2>&1 | tee "$LOG_FILE"

python - "$SUBMISSION_FILE" "$LOG_FILE" "$OOF_FILE" "$RESULT_BUNDLE" <<'PY'
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
