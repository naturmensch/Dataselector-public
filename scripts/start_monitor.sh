#!/usr/bin/env bash
set -euo pipefail

# Wrapper to start the XXL full run monitor safely with a lock and conda env
LOCKFILE="/tmp/xxl_monitor.lock"
LOGFILE="outputs/monitor_startup.log"

mkdir -p "outputs"
# Acquire a non-blocking flock on fd 200
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Another monitor run seems active (lock: $LOCKFILE)"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DATASELECTOR_ENV_NAME:-dataselector}"
PY_CMD=("python")
if [ -x "${ROOT}/scripts/exec_in_env.sh" ]; then
    PY_CMD=("${ROOT}/scripts/exec_in_env.sh" --env "${ENV_NAME}" -- python)
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Run monitor; pass through any args
"${PY_CMD[@]}" scripts/xxl_full_run_monitor.py "$@" >> "$LOGFILE" 2>&1
