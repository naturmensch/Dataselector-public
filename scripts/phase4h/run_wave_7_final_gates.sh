#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave7_final_gates"
PREBLACK_STAMP="${PHASE4H_STAMP_DIR}/${WAVE}.preblack.done"

if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

if [[ "${PHASE4H_FORCE_WAVE:-0}" == "1" ]]; then
  rm -f "${PREBLACK_STAMP}"
fi

if [[ -f "${PREBLACK_STAMP}" ]]; then
  if [[ "${PHASE4H_BLACK_CONFIRMED:-0}" == "1" ]]; then
    mark_done "${WAVE}"
    append_status "${WAVE}" "completed" "Final gates completed after manual black confirmation"
    echo "[phase4h] ${WAVE} done"
    exit 0
  fi

  append_status "${WAVE}" "paused" "Waiting for manual black step confirmation"
  cat <<'MSG'
[phase4h] wave7_final_gates is paused at manual black step.
Run this command manually in your terminal:
  /opt/miniconda3/envs/dataselector/bin/python -m black --check --fast --no-cache dataselector tests scripts docs

Then resume with:
  PHASE4H_BLACK_CONFIRMED=1 scripts/phase4h/run_all.sh --resume-from wave7_final_gates
MSG
  exit 3
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Running final contract/selection gates + quality checks"

{
  echo "[${WAVE}] started $(ts_utc)"
  echo "[${WAVE}] log=${LOG_FILE}"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  "${PYTHON_BIN}" -m pytest -q \
    tests/test_build_new_all_tiles.py \
    tests/test_preselection.py \
    tests/test_spatial_constraint.py \
    tests/test_bootstrap_module.py \
    tests/test_generate_reports_diagnostics.py \
    tests/unit/test_metadata_source_policy.py \
    tests/unit/test_canonical_source_contract.py \
    tests/unit/test_runtime_pass_allowlist.py \
    tests/test_adaptive_pipeline.py \
    tests/test_run_adaptive_pipeline_optuna.py \
    tests/unit/test_city_contract.py -rs

  RUN_FULL_INTEGRATION=1 "${PYTHON_BIN}" -m pytest -q \
    tests/e2e/test_thesis_complete_e2e.py::test_thesis_pipeline_smoke -rs

  "${PYTHON_BIN}" -m ruff check .
  "${PYTHON_BIN}" -m isort --check-only .

  echo "[${WAVE}] pre-black checks finished $(ts_utc)"
} | tee "${LOG_FILE}"

printf "completed_at=%s\n" "$(ts_utc)" >"${PREBLACK_STAMP}"
append_status "${WAVE}" "paused" "Pre-black checks passed; awaiting manual black confirmation"

cat <<'MSG'
[phase4h] wave7_final_gates pre-black checks passed.
Manual step required:
  /opt/miniconda3/envs/dataselector/bin/python -m black --check --fast --no-cache dataselector tests scripts docs

If black is green, resume with:
  PHASE4H_BLACK_CONFIRMED=1 scripts/phase4h/run_all.sh --resume-from wave7_final_gates
MSG

exit 3
