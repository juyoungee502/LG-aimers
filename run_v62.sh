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
if "v61_public_complete_shape" not in metadata.get("model_names", []):
    raise SystemExit(1)
if not Path("outputs/v61_oof_predictions.npz").is_file():
    raise SystemExit(1)
PY
then
    echo "Complete v61 artifacts are missing; rebuilding v61 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v61.sh
fi

{
    "$PYTHON_BIN" research_v62_residual_frontier.py
    "$PYTHON_BIN" promote_v62_residual_frontier.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v62_public_residual_frontier \
        --output submission_v62.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v62.zip -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
known = data.loc[data["season"].isin([2023, 2024])].groupby("pitcher_id").size().idxmax()
matched = data.loc[data["pitcher_id"].eq(known)].groupby(
    ["pitcher_hand", "batter_hand"], sort=False,
).head(1)
other = data.loc[data["pitcher_id"].ne(known)].head(max(0, 5 - len(matched)))
sample = pd.concat([matched, other], ignore_index=True).head(5).drop(columns=["control_success"])
sample["season"] = 2025
sample.to_csv(sys.argv[2], index=False, encoding="utf-8")
PY
    (cd "$smoke_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    "$PYTHON_BIN" - "$smoke_dir/output/submission.csv" <<'PY'
import sys
import pandas as pd
result = pd.read_csv(sys.argv[1])
if len(result) != 5 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v62 smoke test failed")
print("Packaged v62 smoke test passed")
PY
} 2>&1 | tee training_v62.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v62.zip"), Path("training_v62.log"),
    Path("outputs/v62_oof_predictions.npz"),
    Path("research/v62_residual_frontier.json"),
    Path("research/v62_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v62.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v62.zip")
PY

echo "Submit: submission_v62.zip"
echo "Copy: outputs/results_v62.zip"
