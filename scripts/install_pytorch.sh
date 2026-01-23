#!/usr/bin/env bash
set -euo pipefail

# Install PyTorch into the specified conda environment.
# Usage:
#   ./scripts/install_pytorch.sh --env dataselector --cuda auto --yes
# Options for --cuda: auto | none | <version> (e.g. 11.8)

ENV_NAME="dataselector"
CUDA_MODE="auto"
ASSUME_YES=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

print_help() {
    cat <<'HELP'
Usage: install_pytorch.sh [--env NAME] [--cuda auto|none|<ver>] [--yes]

Examples:
  ./scripts/install_pytorch.sh --env dataselector --cuda auto --yes
  ./scripts/install_pytorch.sh --env dataselector --cuda none
  ./scripts/install_pytorch.sh --env dataselector --cuda 11.8 --yes

Behavior:
  --cuda auto  : try to detect NVIDIA GPU; install appropriate cudatoolkit if available, else CPU-only
  --cuda none  : install CPU-only build
  --cuda <ver> : install cudatoolkit=<ver>
HELP
}

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            ENV_NAME="$2"; shift 2;;
        --cuda)
            CUDA_MODE="$2"; shift 2;;
        --yes)
            ASSUME_YES=1; shift;;
        --help|-h)
            print_help; exit 0;;
        *) echo "Unknown arg: $1" >&2; print_help; exit 2;;
    esac
done

# Ensure env exists (create if missing) using canonical wrapper
if [ -f "${ROOT}/scripts/exec_in_env.sh" ]; then
    echo "Ensuring environment '${ENV_NAME}' exists..."
    ${ROOT}/scripts/exec_in_env.sh --env ${ENV_NAME} --create ${ASSUME_YES:+--yes} -- true || {
        echo "ERROR: Could not create or update environment '${ENV_NAME}'" >&2
        exit 1
    }
else
    echo "ERROR: exec_in_env.sh missing; cannot ensure environment exists" >&2
    exit 1
fi

# Decide install target
if [ "$CUDA_MODE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        # Default to a commonly available cudatoolkit version; users should override if different
        CUDA_VER="11.8"
        echo "Detected NVIDIA GPU (nvidia-smi); selecting CUDA toolkit ${CUDA_VER} (can be overridden with --cuda)")
        PKG_TARGET="cudatoolkit=${CUDA_VER}"
    else
        echo "No NVIDIA GPU detected; installing CPU-only build"
        PKG_TARGET="cpuonly"
    fi
elif [ "$CUDA_MODE" = "none" ]; then
    PKG_TARGET="cpuonly"
else
    PKG_TARGET="cudatoolkit=${CUDA_MODE}"
fi

echo "Installing PyTorch (+torchvision) into '${ENV_NAME}' with target '${PKG_TARGET}'"

# Use flexible channel priority to avoid solver deadlocks
export CONDA_CHANNEL_PRIORITY=flexible
conda install -n "${ENV_NAME}" -c pytorch pytorch torchvision ${PKG_TARGET} ${ASSUME_YES:+-y}

# Verify installation
echo "Verifying PyTorch installation..."
${ROOT}/scripts/exec_in_env.sh --env ${ENV_NAME} -- python - <<PY
try:
    import torch
    print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
except Exception as e:
    print('ERROR: could not import torch:', e)
    raise
PY

echo "PyTorch installation finished."
