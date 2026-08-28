#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
VERSION=v62 \
EXPECTED_VERSION=v62_fraction_full \
CORRECTION_WEIGHT=1.0 \
bash run_v60.sh
