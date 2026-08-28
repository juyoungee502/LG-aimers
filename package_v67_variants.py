"""Package three fixed v67 policies from one trained model pair."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VARIANTS = (
    ("v67_tabm_conservative", "submission_v67_conservative.zip", 150., -.1, .3),
    ("v67_tabm_max_gain", "submission_v67_max_gain.zip", 150., -.2, .3),
    ("v67_tabm_distribution", "submission_v67_distribution.zip", 100., -.1, .3),
)


def main():
    path = ROOT / "submit/model/metadata.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    outputs = []
    try:
        for version, filename, threshold, r_weight, f_weight in VARIANTS:
            metadata = json.loads(json.dumps(original))
            metadata["version"] = version
            configuration = metadata["v67_multitask_tabm"]
            configuration.update({
                "threshold": threshold,
                "r_weight": r_weight,
                "f_weight": f_weight,
            })
            path.write_text(json.dumps(metadata), encoding="utf-8")
            output = ROOT / "outputs" / filename
            subprocess.run([
                sys.executable, "build_submission.py",
                "--expected-version", version,
                "--output", str(output),
            ], cwd=ROOT, check=True)
            outputs.append(str(output))
    finally:
        path.write_text(json.dumps(original), encoding="utf-8")
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
