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
if "v58_public_feedback_counterstep" not in metadata.get("model_names", []):
    raise SystemExit(1)
if not Path("outputs/v58_oof_predictions.npz").is_file():
    raise SystemExit(1)
PY
then
    echo "Complete v58 artifacts are missing; rebuilding v58 first."
    PYTHON_BIN="$PYTHON_BIN" bash run_v58.sh
fi

{
    "$PYTHON_BIN" research_v59_public_count_direction.py
    "$PYTHON_BIN" promote_v59_public_exposure.py
    "$PYTHON_BIN" build_submission.py \
        --expected-version v59_public_batter_exposure \
        --output submission_v59.zip

    smoke_dir="$(mktemp -d)"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    unzip -q submission_v59.zip -d "$smoke_dir"
    mkdir -p "$smoke_dir/data" "$smoke_dir/output"
    "$PYTHON_BIN" - data/train.csv "$smoke_dir/data/test.csv" <<'PY'
import sys
import pandas as pd
data = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
known = data.loc[data["season"].isin([2023, 2024])].groupby("batter_id").size()
high_id = known.idxmax()
high = data.loc[data["batter_id"].eq(high_id)].tail(2)
other = data.loc[data["batter_id"].ne(high_id)].head(3)
sample = pd.concat([high, other], ignore_index=True).drop(columns=["control_success"])
sample["season"] = 2025
sample.to_csv(sys.argv[2], index=False, encoding="utf-8")
PY
    (cd "$smoke_dir" && "$OLDPWD/$PYTHON_BIN" script.py)
    "$PYTHON_BIN" - "$smoke_dir/output/submission.csv" <<'PY'
import sys
import pandas as pd
result = pd.read_csv(sys.argv[1])
if len(result) != 5 or not result["control_success"].between(0., 1.).all():
    raise RuntimeError("Packaged v59 smoke test failed")
print("Packaged v59 smoke test passed")
PY
} 2>&1 | tee training_v59.log

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import zipfile
inputs = [
    Path("submission_v59.zip"), Path("training_v59.log"),
    Path("outputs/v59_oof_predictions.npz"),
    Path("research/v59_player_structure.json"),
    Path("research/v59_public_count_direction.json"),
    Path("research/v59_promotion.json"),
]
with zipfile.ZipFile("outputs/results_v59.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in inputs:
        if path.is_file():
            archive.write(path, path.name)
print("Created outputs/results_v59.zip")
PY

echo "Submit: submission_v59.zip"
echo "Copy: outputs/results_v59.zip"
