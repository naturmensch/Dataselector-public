#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave5_golive_policy24"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Starting policy-24 go-live evidence runs (A/B/Hamburg)"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
A_DIR="outputs/thesis_run_A_policy24_${RUN_TAG}"
B_DIR="outputs/thesis_run_B_policy24_${RUN_TAG}"
H_DIR="outputs/thesis_run_hamburg_policy24_${RUN_TAG}"
EVIDENCE_MD="reports_2026-02-09/GO_LIVE_EVIDENCE_POLICY24_${RUN_TAG}.md"

{
  echo "[${WAVE}] started $(ts_utc)"
  echo "[${WAVE}] run_tag=${RUN_TAG}"
  echo "[${WAVE}] log=${LOG_FILE}"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  export RUN_FULL_INTEGRATION="${RUN_FULL_INTEGRATION:-1}"
  export DATASELECTOR_IMAGE_DIR="${DATASELECTOR_IMAGE_DIR:-${REPO_ROOT}/data/images}"

  echo "[${WAVE}] DATASELECTOR_IMAGE_DIR=${DATASELECTOR_IMAGE_DIR}"
  echo "[${WAVE}] RUN_FULL_INTEGRATION=${RUN_FULL_INTEGRATION}"

  "${PYTHON_BIN}" -m dataselector thesis-pipeline \
    --execution-profile thesis_repro \
    --seed 42 \
    --n-samples 24 \
    --validation-seeds 42 \
    --validation-min-distances 28.5 \
    --output-dir "${A_DIR}"

  "${PYTHON_BIN}" -m dataselector thesis-pipeline \
    --execution-profile thesis_repro \
    --seed 42 \
    --n-samples 24 \
    --validation-seeds 42 \
    --validation-min-distances 28.5 \
    --output-dir "${B_DIR}"

  "${PYTHON_BIN}" -m dataselector thesis-pipeline \
    --execution-profile thesis_repro \
    --seed 42 \
    --n-samples 24 \
    --validation-seeds 42 \
    --validation-min-distances 28.5 \
    --hamburg \
    --output-dir "${H_DIR}"

  "${PYTHON_BIN}" scripts/phase4h/build_golive_policy24_report.py \
    --run-a "${A_DIR}" \
    --run-b "${B_DIR}" \
    --run-h "${H_DIR}" \
    --seed 42 \
    --n-samples 24 \
    --min-distance 28.5 \
    --run-tag "${RUN_TAG}" \
    --output-md "${EVIDENCE_MD}" \
    --append-to "reports_2026-02-09/GO_LIVE_EVIDENCE_2026-02-09.md"

  echo "[${WAVE}] evidence_report=${EVIDENCE_MD}"
  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "Policy-24 go-live evidence completed (see ${EVIDENCE_MD}, ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
