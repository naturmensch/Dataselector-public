#!/usr/bin/env bash
# Monitor an experiment session log and append milestone messages to a .milestones.log file
# Usage: ./scripts/monitor_experiment_milestones.sh [--log path_to_session_log]
set -euo pipefail
LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--log <session_log>]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
if [[ -z "$LOG" ]]; then
  # pick latest session.log
  LOG=$(ls -1t outputs/experiments/*.session.log 2>/dev/null | head -n1 || true)
  if [[ -z "$LOG" ]]; then
    echo "No session log found in outputs/experiments/" >&2; exit 1
  fi
fi
MILESTONE_LOG="${LOG%.session.log}.milestones.log"
TMP_SEEN="${LOG%.session.log}.milestones.seen"
mkdir -p "$(dirname "$LOG")"
touch "$MILESTONE_LOG" "$TMP_SEEN"

# Patterns to watch (ordered)
PATTERNS=(
  "PHASE 1" 
  "Phase 1 ABGESCHLOSSEN"
  "Fine sweep" 
  "Pareto.*finished"
  "Optuna optimization finished"
  "Optuna:"
  "Bootstrap finished"
  "BEST BOOTSTRAP CANDIDATE"
  "ADAPTIVE PIPELINE COMPLETE"
  "Report written"
)

# Function to mark and write milestone if new
write_milestone(){
  local text="$1"
  # hash the text -> filename-safe
  local id
  id=$(printf "%s" "$text" | md5sum | awk '{print $1}')
  if ! grep -q "$id" "$TMP_SEEN" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) $text" >> "$MILESTONE_LOG"
    echo "$id" >> "$TMP_SEEN"
  fi
}

# Tail the log and search
# We use grep --line-buffered for immediate reaction
tail -F -n0 "$LOG" | while IFS= read -r line; do
  for pat in "${PATTERNS[@]}"; do
    if echo "$line" | grep -i -E "$pat" >/dev/null 2>&1; then
      write_milestone "$line"
      # Also echo to stdout so user running the monitor can see it
      echo "MILESTONE: $line"
      break
    fi
  done
done
