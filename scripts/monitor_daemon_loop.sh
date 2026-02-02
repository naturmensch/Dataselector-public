#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

LOG="$1"
INTERVAL=${2:-900}
OUTDIR=outputs/experiments
TS_FMT="%Y%m%dT%H%M%SZ"

if [[ -z "$LOG" ]]; then
  echo "Usage: $0 <logfile> [interval_seconds]"; exit 2
fi

mkdir -p "$OUTDIR"

while true; do
  ts=$(date -u +"$TS_FMT")
  out_snapshot="$OUTDIR/monitor_snapshot_${ts}.txt"
  bash ./scripts/monitor_run.sh --log "$LOG" --tail 80 --once > "$out_snapshot" 2>&1 || true
  # also copy optuna_results.csv and pipeline_config to timestamped copies for history
  cp -f outputs/optuna_results.csv "$OUTDIR/optuna_results_${ts}.csv" 2>/dev/null || true
  cp -f outputs/pipeline_config.optuna.yaml "$OUTDIR/pipeline_config_optuna_${ts}.yaml" 2>/dev/null || true
  sleep "$INTERVAL"
done
