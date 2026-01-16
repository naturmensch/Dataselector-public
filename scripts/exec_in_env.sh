#!/usr/bin/env bash
set -euo pipefail

# Exec a command inside a conda/mamba env (mamba preferred) or fallback to local venv or system python.
# Usage:
#   ./scripts/exec_in_env.sh [--env NAME] [--threads N] -- <command> [args...]
# Examples:
#   ./scripts/exec_in_env.sh --env dataselector -- python scripts/run_adaptive_pipeline.py --yes
#   ./scripts/exec_in_env.sh -- python -c "import sys; print(sys.path)"

ENV_NAME=${CONDA_ENV:-dataselector}
THREADS=4

# parse args
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"; shift 2;;
    --threads)
      THREADS="$2"; shift 2;;
    --)
      shift; break;;
    -h|--help)
      echo "Usage: $0 [--env NAME] [--threads N] -- <command> [args...]"; exit 0;;
    *)
      break;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "Error: missing command to run" >&2
  exit 2
fi

# command to run
CMD=("$@")

# export safe thread caps
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS}

# prefer mamba run -> conda run -> local .venv -> direct
if command -v mamba >/dev/null 2>&1; then
  # check env exists
  if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Running in conda env via mamba: $ENV_NAME"
    # Quote the command to exec inside a shell within the env to ensure arguments are forwarded correctly
    sh_cmd=$(printf '%q ' "${CMD[@]}")
    exec mamba run -n "$ENV_NAME" -- bash -lc "$sh_cmd"
  fi
fi
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Running in conda env via conda: $ENV_NAME"
    sh_cmd=$(printf '%q ' "${CMD[@]}")
    exec conda run -n "$ENV_NAME" -- bash -lc "$sh_cmd"
  fi
fi

# fallback: activate local .venv if exists
if [ -f ".venv/bin/activate" ]; then
  echo "Activating local venv .venv and running command"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec "${CMD[@]}"
fi

# final fallback: run directly
echo "No conda env '$ENV_NAME' found and no .venv present — running command directly"
exec "${CMD[@]}"
