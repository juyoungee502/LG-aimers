#!/usr/bin/env bash
set -Eeuo pipefail
VERSION="v25"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"

if [[ ! -f outputs/v24_oof_predictions.npz ]]; then
    echo "v24 OOF is missing; building the v24 base first."
    bash run_v24.sh
fi

required=(
    data/train.csv
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
    echo "[$(date '+%F %T')] Freezing strict temporal portfolio for ${VERSION}"
    python train_v25_temporal_portfolio.py
    python build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v25_strict_temporal_portfolio
    echo "[$(date '+%F %T')] Validation and submission build completed"
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
