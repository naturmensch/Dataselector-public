#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${DATASELECTOR_ENV_NAME:-dataselector}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER=()
if command -v micromamba >/dev/null 2>&1; then
  RUNNER=(micromamba run -n "${ENV_NAME}" --)
elif [ -x "${REPO_ROOT}/scripts/exec_in_env.sh" ]; then
  RUNNER=("${REPO_ROOT}/scripts/exec_in_env.sh" --env "${ENV_NAME}" --)
fi
PHASE4H_OUT="${REPO_ROOT}/outputs/phase4h"
PHASE4H_LOG_DIR="${PHASE4H_OUT}/logs"
PHASE4H_STAMP_DIR="${PHASE4H_OUT}/.stamps"
STATUS_PLAN="${REPO_ROOT}/docs/status/phase4h_scientific_completion_plan_2026-02-09.md"

mkdir -p "${PHASE4H_LOG_DIR}" "${PHASE4H_STAMP_DIR}"

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_file_for() {
  local wave="$1"
  echo "${PHASE4H_LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ)_${wave}.log"
}

stamp_path_for() {
  local wave="$1"
  echo "${PHASE4H_STAMP_DIR}/${wave}.done"
}

has_stamp() {
  local wave="$1"
  local s
  s="$(stamp_path_for "${wave}")"
  [[ -f "${s}" ]]
}

mark_done() {
  local wave="$1"
  local s
  s="$(stamp_path_for "${wave}")"
  printf "completed_at=%s\n" "$(ts_utc)" >"${s}"
}

append_status() {
  local wave="$1"
  local state="$2"
  local details="$3"
  if [[ ${#RUNNER[@]} -gt 0 ]]; then
    "${RUNNER[@]}" python "${SCRIPT_DIR}/update_status.py" \
      --plan-file "${STATUS_PLAN}" \
      --wave "${wave}" \
      --state "${state}" \
      --details "${details}"
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/update_status.py" \
    --plan-file "${STATUS_PLAN}" \
    --wave "${wave}" \
    --state "${state}" \
    --details "${details}"
  fi
}

git_sha_short() {
  git -C "${REPO_ROOT}" rev-parse --short HEAD
}

git_branch_name() {
  git -C "${REPO_ROOT}" branch --show-current
}

print_phase4h_context() {
  local wave="$1"
  if [[ ${#RUNNER[@]} -gt 0 ]]; then
    "${RUNNER[@]}" python - "${REPO_ROOT}" "${wave}" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

root = Path(sys.argv[1])
wave = sys.argv[2]
cfg_path = root / "config" / "pipeline_config.yaml"
csv_path = root / "data" / "new_all_tiles.csv"

def git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=root, text=True).strip()

sha = git(["git", "rev-parse", "--short", "HEAD"])
branch = git(["git", "branch", "--show-current"])

cfg = {}
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

sel = cfg.get("selection", {})
n_samples = sel.get("n_samples")
min_dist = sel.get("min_distance_km")
val_seeds = sel.get("validation_seeds")

rows = None
if csv_path.exists():
    try:
        rows = len(pd.read_csv(csv_path))
    except Exception:
        rows = "read_error"

print(f"[{wave}] branch={branch}")
print(f"[{wave}] sha={sha}")
print(f"[{wave}] config.selection.n_samples={n_samples}")
print(f"[{wave}] config.selection.min_distance_km={min_dist}")
print(f"[{wave}] config.selection.validation_seeds={val_seeds}")
print(f"[{wave}] canonical_rows={rows}")
PY
  else
    "${PYTHON_BIN}" - "${REPO_ROOT}" "${wave}" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

root = Path(sys.argv[1])
wave = sys.argv[2]
cfg_path = root / "config" / "pipeline_config.yaml"
csv_path = root / "data" / "new_all_tiles.csv"

def git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=root, text=True).strip()

sha = git(["git", "rev-parse", "--short", "HEAD"])
branch = git(["git", "branch", "--show-current"])

cfg = {}
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

sel = cfg.get("selection", {})
n_samples = sel.get("n_samples")
min_dist = sel.get("min_distance_km")
val_seeds = sel.get("validation_seeds")

rows = None
if csv_path.exists():
    try:
        rows = len(pd.read_csv(csv_path))
    except Exception:
        rows = "read_error"

print(f"[{wave}] branch={branch}")
print(f"[{wave}] sha={sha}")
print(f"[{wave}] config.selection.n_samples={n_samples}")
print(f"[{wave}] config.selection.min_distance_km={min_dist}")
print(f"[{wave}] config.selection.validation_seeds={val_seeds}")
print(f"[{wave}] canonical_rows={rows}")
PY
  fi
}
