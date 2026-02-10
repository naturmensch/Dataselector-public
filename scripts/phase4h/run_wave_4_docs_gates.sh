#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave4_docs_gates"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Running quick gates and docs consistency checks"

{
  echo "[${WAVE}] started $(ts_utc)"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  "${PYTHON_BIN}" -m pytest -q \
    tests/test_spatial_constraint.py \
    tests/test_bootstrap_module.py \
    tests/test_generate_reports_diagnostics.py \
    tests/unit/test_metadata_source_policy.py \
    tests/unit/test_canonical_source_contract.py \
    tests/unit/test_runtime_pass_allowlist.py -rs

  RUN_FULL_INTEGRATION=1 "${PYTHON_BIN}" -m pytest -q \
    tests/e2e/test_thesis_complete_e2e.py::test_thesis_pipeline_smoke -rs

  "${PYTHON_BIN}" -m ruff check .
  "${PYTHON_BIN}" -m isort --check-only .

  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "Quick docs/gate checks passed (see ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
