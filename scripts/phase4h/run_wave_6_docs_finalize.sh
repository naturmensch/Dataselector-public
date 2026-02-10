#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave6_docs_finalize"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Finalizing docs and evidence-link consistency"

{
  echo "[${WAVE}] started $(ts_utc)"
  echo "[${WAVE}] log=${LOG_FILE}"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  "${PYTHON_BIN}" scripts/phase4h/finalize_docs.py --repo-root "${REPO_ROOT}"

  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "Docs finalized and consistency summary written (see ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
