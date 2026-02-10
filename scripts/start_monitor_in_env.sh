#!/usr/bin/env bash
set -euo pipefail

LOG_OPTUNA=$(ls -1t outputs/runs/optuna_seeded*.log 2>/dev/null | head -n1 || true)
INTERVAL=${1:-900}
if [[ -z "$LOG_OPTUNA" ]]; then
  echo "No optuna log found"; exit 2
fi

# Start the loop and background it
nohup bash ./scripts/monitor_daemon_loop.sh "$LOG_OPTUNA" "$INTERVAL" > outputs/runs/monitor_daemon_loop.out 2>&1 &
echo $!
