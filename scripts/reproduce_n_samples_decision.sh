#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/dataselector/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISTANCE="${DISTANCE:-28.5}"
SEEDS=(${SEEDS:-42 43 44 45 46})
CANDIDATES=(${CANDIDATES:-24 28 32 34 40})
BASE_OUT="${ROOT}/reports_2026-02-09/n_samples"

echo "[reproduce-n-samples] python=${PYTHON_BIN}"
echo "[reproduce-n-samples] distance=${DISTANCE}"
echo "[reproduce-n-samples] seeds=${SEEDS[*]}"
echo "[reproduce-n-samples] candidates=${CANDIDATES[*]}"

cd "${ROOT}"
for n in "${CANDIDATES[@]}"; do
  OUT_DIR="${BASE_OUT}/n_${n}"
  mkdir -p "${OUT_DIR}"
  echo "[reproduce-n-samples] n=${n} -> ${OUT_DIR}"
  "${PYTHON_BIN}" scripts/compare_min_distance_policies.py \
    --metadata-path data/new_all_tiles.csv \
    --distances "${DISTANCE}" \
    --seeds "${SEEDS[@]}" \
    --n-samples "${n}" \
    --output-dir "${OUT_DIR}"
done

echo "[reproduce-n-samples] done"

