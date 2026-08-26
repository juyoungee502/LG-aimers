#!/usr/bin/env bash
set -Eeuo pipefail
VERSION="v16"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"
if [[ ! -f data/train.csv ]]; then
    echo "ERROR: data/train.csv is missing. Run 'git lfs pull' first." >&2
    exit 1
fi
{
    echo "[$(date '+%F %T')] Building frozen pitch-failure prior for ${VERSION}"
    python upgrade_v15_to_v16.py
    python build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v16_pitch_failure_prior
    echo "[$(date '+%F %T')] Upgrade and submission build completed"
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
