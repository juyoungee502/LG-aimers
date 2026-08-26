#!/usr/bin/env bash
set -Eeuo pipefail
VERSION="v18"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"
for required in data/train.csv outputs/v17_oof_predictions.npz submit/model/metadata.json; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: $required is missing." >&2
        exit 1
    fi
done
{
    echo "[$(date '+%F %T')] Training post-break F specialist for ${VERSION}"
    python train_f_regime_specialist.py
    python build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v18_f_regime
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
