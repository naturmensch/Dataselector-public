#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/dataselector/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/reports_2026-02-09"

echo "[reproduce-min-distance] python=${PYTHON_BIN}"
echo "[reproduce-min-distance] output=${OUT_DIR}"

cd "${ROOT}"
"${PYTHON_BIN}" scripts/compare_min_distance_policies.py \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 40 45 \
  --seeds 42 43 44 45 46 \
  --output-dir "${OUT_DIR}"

echo "[reproduce-min-distance] done"

