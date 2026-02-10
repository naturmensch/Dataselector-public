#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave2_distance"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Starting min-distance policy comparison"

{
  echo "[${WAVE}] started $(ts_utc)"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  "${PYTHON_BIN}" scripts/compare_min_distance_policies.py \
    --metadata-path data/new_all_tiles.csv \
    --distances 28.5 40 45 \
    --seeds 42 43 44 45 46 \
    --output-dir reports_2026-02-09

  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "Distance policy comparison completed (see ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
