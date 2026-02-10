#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave3_nsamples"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Starting n-samples candidate sweep"

CANDIDATES=(${N_SAMPLES_CANDIDATES:-24 28 32 34 40})
DISTANCE="${N_SAMPLES_DISTANCE:-28.5}"
SEEDS=(${N_SAMPLES_SEEDS:-42 43 44 45 46})

{
  echo "[${WAVE}] started $(ts_utc)"
  print_phase4h_context "${WAVE}"
  echo "[${WAVE}] candidates=${CANDIDATES[*]}"
  echo "[${WAVE}] distance=${DISTANCE}"
  echo "[${WAVE}] seeds=${SEEDS[*]}"
  cd "${REPO_ROOT}"

  mkdir -p reports_2026-02-09/n_samples

  for n in "${CANDIDATES[@]}"; do
    echo "[${WAVE}] evaluating n_samples=${n}"
    "${PYTHON_BIN}" scripts/compare_min_distance_policies.py \
      --metadata-path data/new_all_tiles.csv \
      --distances "${DISTANCE}" \
      --seeds "${SEEDS[@]}" \
      --n-samples "${n}" \
      --output-dir "reports_2026-02-09/n_samples/n_${n}"
  done

  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "n-samples sweep completed (see ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
