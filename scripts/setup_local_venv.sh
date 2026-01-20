#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_local_venv.sh [VENV_DIR] [PYTHON_EXECUTABLE]
# Examples:
#   ./scripts/setup_local_venv.sh       # creates .venv with system python
#   ./scripts/setup_local_venv.sh .venv python3.11

VENV_DIR=${1:-.venv}
PYTHON=${2:-python3}

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

Done. Activate the environment with:

  source $VENV_DIR/bin/activate

Then you can run project scripts like:

  ./scripts/exec_in_env.sh -- python scripts/run_adaptive_pipeline.py --yes

EOF
