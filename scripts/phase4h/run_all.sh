#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/phase4h/run_all.sh [--resume-from <wave-id>] [--force-wave <wave-id>]...

Examples:
  scripts/phase4h/run_all.sh
  scripts/phase4h/run_all.sh --resume-from wave5_golive_policy24
  scripts/phase4h/run_all.sh --force-wave wave2_distance
  scripts/phase4h/run_all.sh --resume-from wave6_docs_finalize --force-wave wave6_docs_finalize
USAGE
}

RESUME_FROM=""
FORCE_WAVES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-from)
      RESUME_FROM="${2:-}"
      if [[ -z "${RESUME_FROM}" ]]; then
        usage
        exit 2
      fi
      shift 2
      ;;
    --force-wave)
      if [[ -z "${2:-}" ]]; then
        usage
        exit 2
      fi
      FORCE_WAVES+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

is_forced_wave() {
  local wave="$1"
  local forced
  for forced in "${FORCE_WAVES[@]:-}"; do
    if [[ "${forced}" == "${wave}" ]]; then
      return 0
    fi
  done
  return 1
}

waves=(
  "wave1_city:${SCRIPT_DIR}/run_wave_1_city.sh"
  "wave2_distance:${SCRIPT_DIR}/run_wave_2_distance.sh"
  "wave3_nsamples:${SCRIPT_DIR}/run_wave_3_nsamples.sh"
  "wave4_docs_gates:${SCRIPT_DIR}/run_wave_4_docs_gates.sh"
  "wave5_golive_policy24:${SCRIPT_DIR}/run_wave_5_golive_policy24.sh"
  "wave6_docs_finalize:${SCRIPT_DIR}/run_wave_6_docs_finalize.sh"
  "wave7_final_gates:${SCRIPT_DIR}/run_wave_7_final_gates.sh"
)

run_enabled=1
if [[ -n "${RESUME_FROM}" ]]; then
  run_enabled=0
fi

for item in "${waves[@]}"; do
  wave="${item%%:*}"
  cmd="${item#*:}"

  if [[ -n "${RESUME_FROM}" && "${wave}" == "${RESUME_FROM}" ]]; then
    run_enabled=1
  fi

  if [[ "${run_enabled}" -eq 0 ]]; then
    echo "[phase4h] skipping ${wave} (before resume point)"
    continue
  fi

  if has_stamp "${wave}" && ! is_forced_wave "${wave}"; then
    echo "[phase4h] skipping ${wave} (already done: $(stamp_path_for "${wave}"))"
    continue
  fi

  if has_stamp "${wave}" && is_forced_wave "${wave}"; then
    echo "[phase4h] forcing re-run for ${wave} despite stamp"
  fi

  echo "[phase4h] executing ${wave}"
  if is_forced_wave "${wave}"; then
    PHASE4H_FORCE_WAVE=1 "${cmd}"
  else
    "${cmd}"
  fi
done

echo "[phase4h] all configured waves completed"
