#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper (legacy entrypoint): delegates to canonical CLI orchestrator.
# Canonical path:
#   micromamba run -n dataselector -- python -m dataselector thesis-orchestrate ...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="dataselector"

PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --precompute-only)
      PASSTHROUGH_ARGS+=("--precompute-only")
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac

done

echo "[compat] scripts/run_complete_thesis_pipeline.sh delegates to thesis-orchestrate"
echo "[compat] canonical runtime: micromamba run -n ${ENV_NAME} -- ..."

if ! command -v micromamba >/dev/null 2>&1; then
  echo "ERROR: micromamba not found. Install micromamba or use scripts/exec_in_env.sh in compatibility mode." >&2
  exit 2
fi

exec micromamba run -n "$ENV_NAME" -- \
  python -m dataselector thesis-orchestrate \
  "${PASSTHROUGH_ARGS[@]}"
