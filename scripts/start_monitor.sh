#!/usr/bin/env bash
set -euo pipefail

# Wrapper to start the XXL full run monitor safely with a lock and conda env
LOCKFILE="/tmp/xxl_monitor.lock"
LOGFILE="outputs/monitor_startup.log"

mkdir -p "outputs"
# Acquire a non-blocking flock on fd 200
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Another monitor run seems active (lock: $LOCKFILE)"; exit 1; }

# Try to activate conda env if available
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate dataselector || true
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Run monitor; pass through any args
python scripts/xxl_full_run_monitor.py "$@" >> "$LOGFILE" 2>&1
