#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DATASELECTOR_ENV_NAME:-dataselector}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PY_CMD=("${PYTHON_BIN}")
if command -v micromamba >/dev/null 2>&1; then
  PY_CMD=(micromamba run -n "${ENV_NAME}" -- python)
elif [ -x "${ROOT}/scripts/exec_in_env.sh" ]; then
  PY_CMD=("${ROOT}/scripts/exec_in_env.sh" --env "${ENV_NAME}" -- python)
fi
DISTANCE="${DISTANCE:-28.5}"
SEEDS=(${SEEDS:-42 43 44 45 46})
CANDIDATES=(${CANDIDATES:-24 28 32 34 40})
BASE_OUT="${ROOT}/docs/06_REFERENCE/thesis_decision_evidence/n_samples"

echo "[reproduce-n-samples] python=${PY_CMD[*]}"
echo "[reproduce-n-samples] distance=${DISTANCE}"
echo "[reproduce-n-samples] seeds=${SEEDS[*]}"
echo "[reproduce-n-samples] candidates=${CANDIDATES[*]}"

cd "${ROOT}"
for n in "${CANDIDATES[@]}"; do
  OUT_DIR="${BASE_OUT}/n_${n}"
  mkdir -p "${OUT_DIR}"
  echo "[reproduce-n-samples] n=${n} -> ${OUT_DIR}"
  "${PY_CMD[@]}" -m dataselector compare-min-distance-policies \
    --metadata-path data/new_all_tiles.csv \
    --distances "${DISTANCE}" \
    --seeds "${SEEDS[@]}" \
    --n-samples "${n}" \
    --output-dir "${OUT_DIR}"
done

echo "[reproduce-n-samples] done"
