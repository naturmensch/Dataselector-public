#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/create_env.sh [env-name] [python-version] [--force]
ENV_NAME=${1:-dataselector}
PY_VER=${2:-3.11}
FORCE=false
if [[ "${3:-}" == "--force" ]]; then FORCE=true; fi

# Prefer micromamba, then mamba, then conda
PM=conda
if command -v micromamba >/dev/null 2>&1; then
  PM=micromamba
elif command -v mamba >/dev/null 2>&1; then
  PM=mamba
fi

echo "Using package manager: $PM"

ACTIVATE_CMD="conda activate $ENV_NAME"
if [ "$PM" = "micromamba" ]; then
  ACTIVATE_CMD="micromamba activate $ENV_NAME"
fi

if [ "$FORCE" = true ]; then
  echo "Removing existing environment '$ENV_NAME' (force)"
  $PM env remove -n "$ENV_NAME" -y || true
fi

if [ -f environment.yml ]; then
  echo "Creating environment '$ENV_NAME' from environment.yml"
  $PM env create -f environment.yml -n "$ENV_NAME"
else
  echo "No environment.yml found — creating minimal environment '$ENV_NAME'"
  $PM create -n "$ENV_NAME" python="$PY_VER" -y
  echo "Installing core packages from conda-forge and pytorch channels"
  $PM install -n "$ENV_NAME" -c conda-forge pandas=2 numpy scipy scikit-learn umap-learn apricot-select optuna matplotlib seaborn pyyaml tqdm -y
  $PM install -n "$ENV_NAME" -c pytorch cpuonly pytorch torchvision -y
fi

# Install pip extras from requirements file (if present) using the env runner
if [ -f requirements-cpu.txt ]; then
  echo "Installing pip extras from requirements-cpu.txt into '$ENV_NAME' (best-effort)"
  $PM run -n "$ENV_NAME" pip install -r requirements-cpu.txt || true
fi

# Quick smoke check inside the created environment
echo "Running a quick import smoke-test inside '$ENV_NAME'"
$PM run -n "$ENV_NAME" python - <<PY
import importlib,sys
modules=['pandas','numpy','scipy','sklearn','optuna']
missing=[m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    print('Missing:', missing)
    sys.exit(1)
print('Environment smoke test OK')
PY

cat <<EOF

Done. To use the environment run:

  $ACTIVATE_CMD
EOF
