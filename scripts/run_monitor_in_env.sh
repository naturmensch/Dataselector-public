#!/usr/bin/env bash
set -euo pipefail

LOG_OPTUNA=$(ls -1t outputs/runs/optuna_seeded*.log 2>/dev/null | head -n1 || true)
if [[ -z "$LOG_OPTUNA" ]]; then
  echo "No optuna log found"; exit 2
fi

# Run the monitor once and print to stdout
bash ./scripts/monitor_run.sh --log "$LOG_OPTUNA" --tail 80 --once
