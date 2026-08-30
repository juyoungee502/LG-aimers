#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
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
if not Path("outputs/results_v61.zip").is_file():
    raise SystemExit(1)
PY
then
    echo "Complete v61 artifacts are missing; rebuilding v61 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v61.sh
fi

{
    PYTHONWARNINGS="ignore" "$PYTHON_BIN" research_v63_proxy_calibration.py --version 61
    "$PYTHON_BIN" research_v63_train_trend_calibration.py
    "$PYTHON_BIN" promote_v63_train_trend_calibration.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v63_train_trend_calibration \
        --output submission_v63.zip

    smoke_dir="$(mktemp -d)"
    baseline_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir" "$baseline_dir"' EXIT
    unzip -q submission_v63.zip -d "$smoke_dir"
    "$PYTHON_BIN" - outputs/results_v61.zip "$baseline_dir" <<'PY'
import io
from pathlib import Path
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as outer:
    payload = outer.read("submission_v61.zip")
with zipfile.ZipFile(io.BytesIO(payload)) as inner:
    inner.extractall(Path(sys.argv[2]))
PY
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
sample = data.loc[data["season"].eq(2024)].head(5).drop(columns=["control_success"])
sample["season"] = 2025
    sample.to_csv(sys.argv[2], index=False, encoding="utf-8")
PY
    mkdir -p "$baseline_dir/data" "$baseline_dir/output"
    cp "$smoke_dir/data/test.csv" "$baseline_dir/data/test.csv"
    (cd "$baseline_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    (cd "$smoke_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    "$PYTHON_BIN" - \
        "$baseline_dir/output/submission.csv" \
        "$smoke_dir/output/submission.csv" <<'PY'
import sys
import numpy as np
import pandas as pd
baseline = pd.read_csv(sys.argv[1])
result = pd.read_csv(sys.argv[2])
if len(result) != 5 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v63 smoke test failed")
delta = result["control_success"].to_numpy() - baseline["control_success"].to_numpy()
if not np.allclose(delta, -0.0015, atol=1e-12):
    raise RuntimeError(f"Packaged v63 offset mismatch: {delta}")
print("Packaged v63 smoke test passed")
print("Packaged v63-v61 offset parity passed")
PY
} 2>&1 | tee training_v63.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v63.zip"), Path("training_v63.log"),
    Path("outputs/v63_oof_predictions.npz"),
    Path("research/v63_proxy_calibration_v61.json"),
    Path("research/v63_train_trend_calibration.json"),
    Path("research/v63_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v63.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v63.zip")
PY

echo "Submit: submission_v63.zip"
echo "Copy: outputs/results_v63.zip"
