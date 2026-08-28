#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p outputs research

if ! "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("submit/model/metadata.json")
if not path.is_file():
    raise SystemExit(1)
metadata = json.loads(path.read_text(encoding="utf-8"))
required = [
    "catboost_v54_command.cbm", "catboost_v54_overlap.cbm",
    *(f"catboost_v54_recent_{i}.cbm" for i in range(6)),
    *(f"catboost_v54_joint_{i}.cbm" for i in range(3)),
]
if (
    "v54_roster_robust_command" not in metadata.get("model_names", [])
    or not all((path.parent / name).is_file() for name in required)
):
    raise SystemExit(1)
PY
then
    echo "Complete v54 artifacts are missing; rebuilding v54 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v54.sh
fi

{
    "$PYTHON_BIN" research_v56_v54_agreement.py
    "$PYTHON_BIN" research_v56_v55_step.py
    "$PYTHON_BIN" promote_v56_v54_scaling.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v56_v54_regime_scaling \
        --output submission_v56.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v56.zip -d "$smoke_dir"
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
    (cd "$smoke_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    "$PYTHON_BIN" - "$smoke_dir/output/submission.csv" <<'PY'
import sys
import pandas as pd
result = pd.read_csv(sys.argv[1])
if len(result) != 5 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v56 smoke test failed")
print("Packaged v56 smoke test passed")
PY
} 2>&1 | tee training_v56.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v56.zip"), Path("training_v56.log"),
    Path("outputs/v56_oof_predictions.npz"),
    Path("research/v56_v54_agreement.json"),
    Path("research/v56_v55_step.json"),
]
with zipfile.ZipFile("outputs/results_v56.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v56.zip")
PY

echo "Copy: outputs/results_v56.zip"
