#!/usr/bin/env bash
set -euo pipefail

# BEST PRACTICE: Use conda-lock for reproducible builds
# Usage: ./scripts/setup_local_venv.sh [ENV_NAME|VENV_DIR] [PYTHON_EXECUTABLE]
# Examples:
#   ./scripts/setup_local_venv.sh       # tries mamba/conda first, falls back to .venv
#   ./scripts/setup_local_venv.sh dataselector     # mamba create -n dataselector
#   ./scripts/setup_local_venv.sh .venv python3.11 # venv fallback

ENV_NAME=${1:-.venv}
PYTHON=${2:-python3}

# Try conda-lock first (reproducible)
if command -v micromamba &> /dev/null && [ -f locks/conda-lock-linux-64.lock ]; then
  echo "✓ Using conda-lock (reproducible build)"
  micromamba create -n "$ENV_NAME" --file locks/conda-lock-linux-64.lock -y

  if [ -f requirements-cpu.txt ]; then
    echo "Installing CPU-only PyTorch from requirements-cpu.txt..."
    micromamba run -n "$ENV_NAME" pip install -r requirements-cpu.txt
  fi

  micromamba run -n "$ENV_NAME" pip install -e .

  cat <<EOF

Done. Activate the environment with:

  micromamba activate $ENV_NAME

Then you can run project scripts like:

  ./scripts/exec_in_env.sh -- python scripts/run_adaptive_pipeline.py --yes

EOF

elif command -v mamba &> /dev/null && [ -f locks/conda-lock-linux-64.lock ]; then
  echo "✓ Using conda-lock (reproducible build)"
  mamba create -n "$ENV_NAME" --file locks/conda-lock-linux-64.lock -y

  if [ -f requirements-cpu.txt ]; then
    echo "Installing CPU-only PyTorch from requirements-cpu.txt..."
    mamba run -n "$ENV_NAME" pip install -r requirements-cpu.txt
  fi

  mamba run -n "$ENV_NAME" pip install -e .
  
  cat <<EOF

Done. Activate the environment with:

  mamba activate $ENV_NAME

Then you can run project scripts like:

  ./scripts/exec_in_env.sh -- python scripts/run_adaptive_pipeline.py --yes

EOF

elif command -v conda &> /dev/null && [ -f locks/conda-lock-linux-64.lock ]; then
  echo "✓ Using conda-lock (reproducible build)"
  conda create -n "$ENV_NAME" --file locks/conda-lock-linux-64.lock -y

  if [ -f requirements-cpu.txt ]; then
    echo "Installing CPU-only PyTorch from requirements-cpu.txt..."
    conda run -n "$ENV_NAME" pip install -r requirements-cpu.txt
  fi

  conda run -n "$ENV_NAME" pip install -e .
  
  cat <<EOF

Done. Activate the environment with:

  conda activate $ENV_NAME

Then you can run project scripts like:

  ./scripts/exec_in_env.sh -- python scripts/run_adaptive_pipeline.py --yes

EOF

else
  # Fallback: venv + pip (WARNING: not reproducible)
  echo "⚠ Warning: mamba/conda not found. Falling back to venv + pip (NOT reproducible)"
  echo "  For reproducible builds, install mamba/conda and rerun this script."
  
  VENV_DIR="$ENV_NAME"
  echo "Creating virtual environment at: $VENV_DIR using interpreter: $PYTHON"
  $PYTHON -m venv "$VENV_DIR"
  
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip

  # install from cpu requirements if available, else fallback to generic requirements
  if [ -f requirements-cpu.txt ]; then
    pip install -r requirements-cpu.txt || true
  elif [ -f requirements.txt ]; then
    pip install -r requirements.txt || true
  fi

  # Install the package in editable mode so console-scripts / imports work from the venv
  pip install -e .

  cat <<EOF

Done (venv fallback). Activate the environment with:

  source $VENV_DIR/bin/activate

Then you can run project scripts like:

  ./scripts/exec_in_env.sh -- python scripts/run_adaptive_pipeline.py --yes

EOF
fi
