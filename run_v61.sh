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
if "v60_public_hand_shape" not in metadata.get("model_names", []):
    raise SystemExit(1)
if not Path("outputs/v60_oof_predictions.npz").is_file():
    raise SystemExit(1)
PY
then
    echo "Complete v60 artifacts are missing; rebuilding v60 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v60.sh
fi

{
    "$PYTHON_BIN" research_v61_complete_shape.py
    "$PYTHON_BIN" promote_v61_complete_shape.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v61_public_complete_shape \
        --output submission_v61.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v61.zip -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
known_batter = data.loc[data["season"].isin([2023, 2024])].groupby("batter_id").size().idxmax()
known_pitcher = data.loc[data["season"].isin([2023, 2024])].groupby("pitcher_id").size().idxmax()
matched = data.loc[
    data["batter_id"].eq(known_batter) | data["pitcher_id"].eq(known_pitcher)
].head(4)
other = data.loc[
    data["batter_id"].ne(known_batter) & data["pitcher_id"].ne(known_pitcher)
].head(max(0, 5 - len(matched)))
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
    raise RuntimeError("Packaged v61 smoke test failed")
print("Packaged v61 smoke test passed")
PY
} 2>&1 | tee training_v61.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v61.zip"), Path("training_v61.log"),
    Path("outputs/v61_oof_predictions.npz"),
    Path("research/v61_complete_shape.json"),
    Path("research/v61_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v61.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        archive.write(path, path.name)
print("Created outputs/results_v61.zip")
PY

echo "Submit: submission_v61.zip"
echo "Copy: outputs/results_v61.zip"
