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
OUT_DIR="${ROOT}/docs/06_REFERENCE/thesis_decision_evidence"

echo "[reproduce-min-distance] python=${PY_CMD[*]}"
echo "[reproduce-min-distance] output=${OUT_DIR}"

cd "${ROOT}"
"${PY_CMD[@]}" -m dataselector compare-min-distance-policies \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 40 45 \
  --seeds 42 43 44 45 46 \
  --output-dir "${OUT_DIR}"

echo "[reproduce-min-distance] done"
