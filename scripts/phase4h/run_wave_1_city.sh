#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave1_city"
if has_stamp "${WAVE}" && [[ "${PHASE4H_FORCE_WAVE:-0}" != "1" ]]; then
  echo "[phase4h] ${WAVE} already done: $(stamp_path_for "${WAVE}")"
  exit 0
fi

LOG_FILE="$(log_file_for "${WAVE}")"
append_status "${WAVE}" "running" "Starting canonical city rebuild and contract checks"

{
  echo "[${WAVE}] started $(ts_utc)"
  echo "[${WAVE}] python=${PYTHON_BIN}"
  echo "[${WAVE}] log=${LOG_FILE}"
  print_phase4h_context "${WAVE}"

  cd "${REPO_ROOT}"

  "${PYTHON_BIN}" -m dataselector build-tiles \
    --image-dir data/images \
    --out data/new_all_tiles.csv \
    --name-source-csv data/KDR100_foliage_with_files_epsg3857.csv \
    --city-overrides data/city_overrides.csv

  "${PYTHON_BIN}" - <<'PY'
import pandas as pd

df = pd.read_csv("data/new_all_tiles.csv")
city_non_empty = int((df["city"].fillna("").astype(str).str.strip() != "").sum())
source_non_empty = int((df["city_source"].fillna("").astype(str).str.strip() != "").sum())
rows = len(df)
print(f"[wave1_city] rows={rows}")
print(f"[wave1_city] city_non_empty={city_non_empty}")
print(f"[wave1_city] city_source_non_empty={source_non_empty}")
print(f"[wave1_city] hamburg_rows={(df['city'].astype(str).str.lower() == 'hamburg').sum()}")
print(f"[wave1_city] kiel_rows={(df['city'].astype(str).str.lower() == 'kiel').sum()}")
PY

  "${PYTHON_BIN}" -m pytest -q \
    tests/test_build_new_all_tiles.py \
    tests/test_preselection.py \
    tests/unit/test_city_contract.py -rs

  echo "[${WAVE}] finished $(ts_utc)"
} | tee "${LOG_FILE}"

mark_done "${WAVE}"
append_status "${WAVE}" "completed" "Canonical city contract rebuilt and tests green (see ${LOG_FILE})"
echo "[phase4h] ${WAVE} done"
