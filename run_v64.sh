#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TASK_TYPE="${TASK_TYPE:-GPU}"
DEVICES="${DEVICES:-0}"
mkdir -p outputs research

if ! "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("submit/model/metadata.json")
if not path.is_file() or not Path("outputs/v61_oof_predictions.npz").is_file():
    raise SystemExit(1)
metadata = json.loads(path.read_text(encoding="utf-8"))
if "v61_public_complete_shape" not in metadata.get("model_names", []):
    raise SystemExit(1)
PY
then
    echo "Complete v61 artifacts are missing; rebuilding v61 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v61.sh
fi

{
    "$PYTHON_BIN" research_v64_dynamic_pitcher_state.py
    "$PYTHON_BIN" research_v64_public_f_regime.py \
        --task-type "$TASK_TYPE" --devices "$DEVICES"
    "$PYTHON_BIN" train_v64_public_transfer.py \
        --task-type "$TASK_TYPE" --devices "$DEVICES"
    "$PYTHON_BIN" build_submission.py \
        --expected-version v64_public_method_transfer \
        --output submission_v64.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v64.zip -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
regular = data.loc[data["game_type"].eq("R")].tail(3)
futures = data.loc[data["game_type"].eq("F")].tail(3)
cold = data.loc[data["asof_pitcher_n"].eq(0)].head(2)
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
if len(result) < 6 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v64 smoke test failed")
print("Packaged v64 smoke test passed")
PY
} 2>&1 | tee training_v64.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v64.zip"), Path("training_v64.log"),
    Path("outputs/v64_oof_predictions.npz"),
    Path("research/v64_dynamic_pitcher_state.json"),
    Path("research/v64_public_f_regime.json"),
    Path("research/v64_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v64.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v64.zip")
PY

echo "Submit: submission_v64.zip"
echo "Copy: outputs/results_v64.zip"
