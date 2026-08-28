#!/usr/bin/env bash
set -Eeuo pipefail

VERSION="v54"
LOG_FILE="training_${VERSION}.log"
SUBMISSION_FILE="submission_${VERSION}.zip"
RESULT_DIR="outputs"
OOF_FILE="${RESULT_DIR}/${VERSION}_oof_predictions.npz"
ROSTER_REPORT="research/v53_roster_stability.json"
RESULT_BUNDLE="${RESULT_DIR}/results_${VERSION}.zip"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" research

if [[ ! -f outputs/v38_oof_predictions.npz ]]; then
    echo "v38 OOF is missing; building the v38 base first."
    bash run_v38.sh
fi

required=(
    data/train.csv
    outputs/v38_oof_predictions.npz
    research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz
    research/v35_lowcard_direct_hl2_s3_2024.npz
    submit/model/metadata.json
)
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $path is missing." >&2
        exit 1
    fi
done

{
    if [[ ! -f research/v43_multiclass_hl2_s1_2024.npz ]]; then
        "$PYTHON_BIN" research_v43_multiclass_failure.py \
            --valid-year 2024 --n-seeds 1 --half-life 2 \
            --task-type GPU --devices 0
    fi
    if [[ ! -f research/v45_overlap_hl2_2024.npz ]]; then
        "$PYTHON_BIN" research_v45_overlap_correction.py \
            --valid-year 2024 --half-life 2 --task-type GPU --devices 0
    fi
    if [[ ! -f research/v48_regime_command_s3_2024.npz ]]; then
        "$PYTHON_BIN" research_v48_regime_command.py \
            --n-seeds 3 --task-type GPU --devices 0
    fi
    if [[ ! -f research/v49_regime_multiclass_complexity_2024.npz ]]; then
        "$PYTHON_BIN" research_v49_regime_multiclass_complexity.py \
            --depths 6 --checkpoints 1000 --n-seeds 3 \
            --task-type GPU --devices 0
    fi
    if [[ ! -f research/v52_pitch_command_joint_s3_2024.npz ]]; then
        "$PYTHON_BIN" research_v52_pitch_command_joint.py \
            --valid-year 2024 --n-seeds 3 \
            --modes history history_no_team \
            --task-type GPU --devices 0
    fi
    echo "[$(date '+%F %T')] Auditing roster and team turnover"
    "$PYTHON_BIN" research_v53_roster_stability.py
    echo "[$(date '+%F %T')] Training roster-robust ensemble for ${VERSION}"
    "$PYTHON_BIN" train_v54_roster_robust.py --task-type GPU --devices 0
    "$PYTHON_BIN" build_submission.py --output "$SUBMISSION_FILE" \
        --expected-version v54_roster_robust_command

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q "$SUBMISSION_FILE" -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd

data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
sample = pd.concat([
    data.loc[data["game_type"].eq("R")].head(3),
    data.loc[data["game_type"].eq("F")].head(2),
], ignore_index=True).drop(columns=["control_success"])
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
    raise RuntimeError("Packaged v54 smoke test failed")
print(f"Packaged smoke test passed: rows={len(result)}")
PY
    echo "[$(date '+%F %T')] GPU training, validation, build, and smoke test completed"
} 2>&1 | tee "$LOG_FILE"

"$PYTHON_BIN" - \
    "$SUBMISSION_FILE" "$LOG_FILE" "$OOF_FILE" "$ROSTER_REPORT" \
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
