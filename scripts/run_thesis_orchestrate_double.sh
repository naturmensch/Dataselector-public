#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_thesis_orchestrate_double.sh [options]

Options:
  --single-run                 Run only tag A (instead of A and B).
  --probe-seconds N            Use timeout N seconds per run.
  --autoscale-trials N         Override autoscale trials (for quick probe runs).
  --autoscale-stages "LIST"    Override autoscale stages (e.g. "27 34 40 54").
  --build-splits MODE          MODE in {true,false,auto} (default: false).
  --with-splits                Shortcut for --build-splits true.
  --no-splits                  Shortcut for --build-splits false.
  --anchor-tile NAME           Pre-select an anchor tile (exports DATASELECTOR_ANCHOR_TILE).
  --hamburg                    Shortcut for --anchor-tile "Hamburg".
  -h, --help                   Show this help.

Environment overrides:
  BASE_TS, OUTPUT_ROOT, CONFIG_PATH, EXECUTION_PROFILE, SEED,
  CACHE_MODE, STRICT_EVIDENCE_ROOT, STRICT_REAL_DATA, BUILD_SPLITS,
  SPLIT_SEED, TILE_EXCLUSION_POLICY, SPLIT_POLICY, LEAKAGE_BUFFER_KM,
  XDG_CACHE_HOME, TORCH_HOME.
EOF
}

SINGLE_RUN=false
PROBE_SECONDS=0
AUTOSCALE_TRIALS=""
AUTOSCALE_STAGES=""
BUILD_SPLITS_OVERRIDE=""
ANCHOR_TILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single-run)
      SINGLE_RUN=true
      shift
      ;;
    --probe-seconds)
      PROBE_SECONDS="${2:-}"
      if [[ -z "${PROBE_SECONDS}" ]]; then
        echo "ERROR: --probe-seconds requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --autoscale-trials)
      AUTOSCALE_TRIALS="${2:-}"
      if [[ -z "${AUTOSCALE_TRIALS}" ]]; then
        echo "ERROR: --autoscale-trials requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --autoscale-stages)
      AUTOSCALE_STAGES="${2:-}"
      if [[ -z "${AUTOSCALE_STAGES}" ]]; then
        echo "ERROR: --autoscale-stages requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --build-splits)
      BUILD_SPLITS_OVERRIDE="${2:-}"
      if [[ -z "${BUILD_SPLITS_OVERRIDE}" ]]; then
        echo "ERROR: --build-splits requires a value" >&2
        exit 2
      fi
      case "${BUILD_SPLITS_OVERRIDE}" in
        true|false|auto) ;;
        *)
          echo "ERROR: --build-splits must be one of: true, false, auto" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --with-splits)
      BUILD_SPLITS_OVERRIDE="true"
      shift
      ;;
    --no-splits)
      BUILD_SPLITS_OVERRIDE="false"
      shift
      ;;
    --anchor-tile)
      ANCHOR_TILE="${2:-}"
      if [[ -z "${ANCHOR_TILE}" ]]; then
        echo "ERROR: --anchor-tile requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --hamburg)
      ANCHOR_TILE="Hamburg"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;

    *)
      echo "ERROR: unknown argument '$1'" >&2
      usage
      exit 2
      ;;
  esac
done

BASE_TS="${BASE_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs}"
CONFIG_PATH="${CONFIG_PATH:-config/pipeline_config.yaml}"
EXECUTION_PROFILE="${EXECUTION_PROFILE:-thesis_repro}"
SEED="${SEED:-42}"
CACHE_MODE="${CACHE_MODE:-read_write}"
STRICT_EVIDENCE_ROOT="${STRICT_EVIDENCE_ROOT:-run_dir}"
STRICT_REAL_DATA="${STRICT_REAL_DATA:-true}"
BUILD_SPLITS="${BUILD_SPLITS:-false}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TILE_EXCLUSION_POLICY="${TILE_EXCLUSION_POLICY:-config/tile_exclusion_policy.yaml}"
SPLIT_POLICY="${SPLIT_POLICY:-config/spatial_split_policy.yaml}"
LEAKAGE_BUFFER_KM="${LEAKAGE_BUFFER_KM:-auto}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mamba-cache}"
TORCH_HOME="${TORCH_HOME:-${HOME:-}/.cache/torch}"

if [[ -n "${BUILD_SPLITS_OVERRIDE}" ]]; then
  BUILD_SPLITS="${BUILD_SPLITS_OVERRIDE}"
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${ANCHOR_TILE}" ]]; then
  echo "Applying anchor tile constraint: ${ANCHOR_TILE}"
  export DATASELECTOR_ANCHOR_TILE="${ANCHOR_TILE}"
fi

if [[ "${SINGLE_RUN}" == "true" ]]; then
  tags=("A")
else
  tags=("A" "B")
fi

for tag in "${tags[@]}"; do
  run_dir="${OUTPUT_ROOT}/thesis_orchestrate_full_${BASE_TS}_${tag}"
  log="${OUTPUT_ROOT}/thesis_orchestrate_full_${BASE_TS}_${tag}.log"

  echo "=== START ${tag} ==="
  echo "RUN_DIR=${run_dir}"
  echo "LOG=${log}"
  echo "BUILD_SPLITS=${BUILD_SPLITS}"
  if [[ -n "${AUTOSCALE_STAGES}" ]]; then
    echo "AUTOSCALE_STAGES=${AUTOSCALE_STAGES}"
  else
    echo "AUTOSCALE_STAGES=<core default policy>"
  fi

  cmd=(
    micromamba run -n dataselector
    python -u -m dataselector thesis-orchestrate
    --config "${CONFIG_PATH}"
    --output-dir "${run_dir}"
    --execution-profile "${EXECUTION_PROFILE}"
    --seed "${SEED}"
    --cache-mode "${CACHE_MODE}"
    --strict-evidence-root "${STRICT_EVIDENCE_ROOT}"
    --strict-real-data "${STRICT_REAL_DATA}"
    --split-seed "${SPLIT_SEED}"
    --tile-exclusion-policy "${TILE_EXCLUSION_POLICY}"
    --split-policy "${SPLIT_POLICY}"
    --leakage-buffer-km "${LEAKAGE_BUFFER_KM}"
  )

  cmd+=(--build-splits "${BUILD_SPLITS}")

  if [[ -n "${AUTOSCALE_TRIALS}" ]]; then
    cmd+=(--autoscale-trials "${AUTOSCALE_TRIALS}")
  fi
  if [[ -n "${AUTOSCALE_STAGES}" ]]; then
    read -r -a stage_values <<< "${AUTOSCALE_STAGES}"
    cmd+=(--autoscale-stages "${stage_values[@]}")
  fi

  set +e
  {
    if [[ "${PROBE_SECONDS}" -gt 0 ]]; then
      timeout "${PROBE_SECONDS}s" \
        env XDG_CACHE_HOME="${XDG_CACHE_HOME}" TORCH_HOME="${TORCH_HOME}" PYTHONUNBUFFERED=1 \
        "${cmd[@]}"
    else
      env XDG_CACHE_HOME="${XDG_CACHE_HOME}" TORCH_HOME="${TORCH_HOME}" PYTHONUNBUFFERED=1 \
        "${cmd[@]}"
    fi
  } 2>&1 \
    | stdbuf -o0 tr '\r' '\n' \
    | awk 'NF { print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush(); }' \
    | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e

  echo "EXIT_CODE_${tag}=${rc}"

  if [[ "${PROBE_SECONDS}" -gt 0 ]]; then
    if [[ "${rc}" -eq 124 ]]; then
      echo "Probe for ${tag}: startup OK (timeout reached intentionally)."
    elif [[ "${rc}" -ne 0 ]]; then
      echo "Probe for ${tag} failed with exit code ${rc}." >&2
      exit "${rc}"
    fi
  else
    if [[ "${rc}" -ne 0 ]]; then
      echo "Run ${tag} failed. Stop." >&2
      exit "${rc}"
    fi
  fi
done

if [[ "${PROBE_SECONDS}" -gt 0 ]]; then
  echo "Probe completed: ${BASE_TS}"
else
  echo "Doppelrun abgeschlossen: ${BASE_TS}"
fi
