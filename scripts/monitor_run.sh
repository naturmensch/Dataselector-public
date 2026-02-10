#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Monitor script: prints concise Optuna/Run summary periodically
# Usage:
#   ./scripts/monitor_run.sh --log path/to/run.session.log --interval 300 [--tail 30] [--once] [--filter 'regex']

LOG=""
INTERVAL=300
TAIL=20
ONCE=0
FILTER=""

usage() {
  cat <<EOF
Usage: $0 --log <path> [--interval N] [--tail N] [--once] [--filter '<regex>']

Examples:
  $0 --log outputs/runs/run_adaptive_20260116T150717Z.session.log --interval 300
  $0 --log outputs/optuna_n_samples_range.log --interval 60 --tail 50
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) LOG="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --tail) TAIL="$2"; shift 2;;
    --once) ONCE=1; shift;;
    --filter) FILTER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$LOG" ]]; then
  echo "Error: --log is required"; usage; exit 2
fi

if [[ ! -f "$LOG" ]]; then
  echo "Warning: log file does not exist yet: $LOG"; echo "Proceeding — the file may be created by the running pipeline.";
fi

print_summary() {
  echo "\n===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

  # Optuna summary (if available)
  if [[ -f "outputs/optuna_results.csv" ]]; then
    echo "-- Optuna summary (outputs/optuna_results.csv) --"
    python - <<'PY'
import sys, pandas as pd
try:
    df=pd.read_csv('outputs/optuna_results.csv')
    best=df.loc[df['value'].idxmax()]
    print(f"Best trial: number={int(best['number'])}, value={float(best['value']):.6f}, n_samples={int(best.get('params_n_samples')) if 'params_n_samples' in best.index else best.get('user_attrs_n_samples','n/a')}, min_distance={best.get('params_min_distance_km','n/a')}")
    print('Top-5 (value, number, n_samples, min_distance):')
    print(df.sort_values('value', ascending=False).head(5)[['number','value','params_n_samples','params_min_distance_km']].to_string(index=False))
except Exception as e:
    print('Could not read outputs/optuna_results.csv:', e)
PY
  else
    echo "No outputs/optuna_results.csv found yet"
  fi

  # Print Optuna config pre-selection (if available)
  if [[ -f "outputs/pipeline_config.optuna.yaml" ]]; then
    echo "-- Optuna config (outputs/pipeline_config.optuna.yaml) --"
    python - <<'PY'
import yaml
try:
    cfg=yaml.safe_load(open('outputs/pipeline_config.optuna.yaml'))
    pre=cfg.get('selection',{}).get('pre_selected_names',None)
    print('Optuna pre-selected names:', pre)
except Exception as e:
    print('Could not read optuna config:', e)
PY
  fi

  # Tail the run log
  if [[ -f "$LOG" ]]; then
    echo "\n-- Last $TAIL lines from: $LOG --"
    if [[ -n "$FILTER" ]]; then
      tail -n "$TAIL" "$LOG" | grep -E --line-buffered "$FILTER" || true
    else
      tail -n "$TAIL" "$LOG" || true
    fi
  else
    echo "Log not found: $LOG"
  fi
}

# Main loop
while true; do
  print_summary
  if [[ "$ONCE" -eq 1 ]]; then
    break
  fi
  sleep "$INTERVAL"
done

exit 0
