#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs logs submit/model
.venv/bin/python train_v67_multitask.py 2>&1 | tee logs/v67_train.log
.venv/bin/python package_v67_variants.py 2>&1 | tee logs/v67_package.log
sha256sum outputs/submission_v67_*.zip | tee outputs/v67_sha256.txt
zip -j -q outputs/results_v67.zip \
  outputs/submission_v67_conservative.zip \
  outputs/submission_v67_max_gain.zip \
  outputs/submission_v67_distribution.zip \
  outputs/v67_sha256.txt logs/v67_train.log logs/v67_package.log
echo "Created outputs/results_v67.zip"
