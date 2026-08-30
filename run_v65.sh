#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TASK_TYPE="${TASK_TYPE:-GPU}"
DEVICES="${DEVICES:-0}"
BOOTSTRAP="${BOOTSTRAP:-20000}"
mkdir -p outputs research

if ! "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("submit/model/metadata.json")
if not path.is_file() or not Path("outputs/v64_oof_predictions.npz").is_file():
    raise SystemExit(1)
metadata = json.loads(path.read_text(encoding="utf-8"))
if metadata.get("version") != "v64_public_method_transfer":
    raise SystemExit(1)
PY
then
    echo "Complete v64 artifacts are missing or stale; rebuilding v64 first."
    PYTHON_BIN="$PYTHON_BIN" TASK_TYPE="$TASK_TYPE" DEVICES="$DEVICES" \
        bash run_v64.sh
fi

{
    "$PYTHON_BIN" research_v65_prediction_gap_meta.py \
        --bootstrap "$BOOTSTRAP"
    "$PYTHON_BIN" train_v65_prediction_gap.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v65_prediction_gap_meta \
        --output submission_v65.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v65.zip -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
regular = data.loc[data["game_type"].eq("R")].tail(4)
futures = data.loc[data["game_type"].eq("F")].tail(4)
cold = data.loc[
    data["asof_pitcher_n"].eq(0) | data["asof_batter_n"].eq(0)
].head(3)
sample = pd.concat([cold, regular, futures], ignore_index=True)
sample = sample.drop_duplicates("row_id").drop(columns=["control_success"])
sample["season"] = 2025
sample.to_csv(sys.argv[2], index=False, encoding="utf-8")
PY
    (cd "$smoke_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    "$PYTHON_BIN" - "$smoke_dir/output/submission.csv" <<'PY'
import sys
import pandas as pd
result = pd.read_csv(sys.argv[1])
if len(result) < 8 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v65 smoke test failed")
if not result["control_success"].notna().all():
    raise RuntimeError("Packaged v65 smoke test produced missing predictions")
print("Packaged v65 smoke test passed")
PY
} 2>&1 | tee training_v65.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v65.zip"), Path("training_v65.log"),
    Path("outputs/v65_oof_predictions.npz"),
    Path("research/v65_prediction_gap_meta.json"),
    Path("research/v65_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v65.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v65.zip")
PY

echo "Submit: submission_v65.zip"
echo "Copy: outputs/results_v65.zip"
