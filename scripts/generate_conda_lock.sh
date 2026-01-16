#!/usr/bin/env bash
set -euo pipefail

# Generate conda-lock files for specified platforms (default: linux-64)
# Usage: ./scripts/generate_conda_lock.sh --platform linux-64 --platform osx-64

PLATFORMS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p|--platform)
      PLATFORMS+=("$2"); shift 2;;
    -h|--help)
      echo "Usage: $0 [-p|--platform <platform>]..."; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [ ${#PLATFORMS[@]} -eq 0 ]; then
  PLATFORMS=(linux-64)
fi

if [ ! -f environment.yml ]; then
  echo "error: environment.yml not found in repo root" >&2
  exit 1
fi

# Prefer mamba/conda installation of conda-lock
if command -v mamba >/dev/null 2>&1; then
  if ! command -v conda-lock >/dev/null 2>&1; then
    echo "Installing conda-lock via mamba (conda-forge)"
    mamba install -n base -c conda-forge conda-lock -y || true
  fi
elif command -v conda >/dev/null 2>&1; then
  if ! command -v conda-lock >/dev/null 2>&1; then
    echo "Installing conda-lock via conda (conda-forge)"
    conda install -n base -c conda-forge conda-lock -y || true
  fi
else
  echo "Neither mamba nor conda found — trying pip install conda-lock"
  python -m pip install --upgrade pip
  pip install conda-lock || true
fi

if ! command -v conda-lock >/dev/null 2>&1; then
  echo "conda-lock not available after install attempts; aborting" >&2
  exit 1
fi

mkdir -p locks
for p in "${PLATFORMS[@]}"; do
  out="locks/conda-lock-${p}.yml"
  echo "Generating conda-lock for platform: $p -> $out"
  conda-lock -f environment.yml -p "$p" -o "$out"
done

echo "Generated lockfiles in: $(pwd)/locks"
