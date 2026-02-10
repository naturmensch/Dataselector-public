#!/usr/bin/env bash
# Simple watcher to tail all .log files for a given run directory
# Features:
#  - If no run_dir provided, picks the most recent outputs/runs/run_*
#  - Optional --filter <regex> to only show matching lines (live)
#  - Optional --show-proc to attempt showing related PID/CPU/MEM (no extra deps)
# Usage:
#  ./scripts/watch_experiment.sh [run_dir] [--filter '<regex>'] [--show-proc]

set -euo pipefail
IFS=$'\n\t'

# Default args
FILTER_REGEX=""
SHOW_PROC=0
TAIL_LINES=200

usage() {
  cat <<EOF
Usage: $0 [run_dir] [--filter '<regex>'] [--show-proc] [--lines N]

If run_dir is omitted the latest directory matching 'outputs/runs/run_*' is used.
--filter : live grep regex (use quotes, e.g. --filter 'FAILED|Traceback')
--show-proc : attempt to show PID/CPU/MEM for related processes (uses pgrep/ps, no new deps)
--lines N : how many lines to show initially (default: ${TAIL_LINES})
EOF
}

# Parse args (simple)
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)
      FILTER_REGEX="$2"; shift 2;;
    --show-proc)
      SHOW_PROC=1; shift;;
    --lines)
      TAIL_LINES="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    --*)
      echo "Unknown option: $1"; usage; exit 1;;
    *)
      positional+=("$1"); shift;;
  esac
done

# Determine run dir
if [ ${#positional[@]} -eq 0 ]; then
  RUN_DIR=$(ls -td outputs/runs/run_* 2>/dev/null | head -n1 || true)
  if [ -z "$RUN_DIR" ]; then
    echo "No run directories found under outputs/runs/"
    exit 1
  fi
else
  RUN_DIR="${positional[0]}"
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "Run dir not found: $RUN_DIR"
  exit 1
fi

LOGS=("$RUN_DIR"/*.log)
if [ ${#LOGS[@]} -eq 0 ]; then
  echo "No log files found in $RUN_DIR"
  exit 0
fi

# Print what we'll follow
echo "==> Following logs in: $RUN_DIR <=="
for f in "$RUN_DIR"/*.log; do
  echo "  - $f"
done

# If requested, try to find related processes (no external deps)
if [ "$SHOW_PROC" -eq 1 ]; then
  echo "\nAttempting to locate related processes (ps/pgrep used)..."
  # Candidate patterns: run_full_experiment.sh, run_adaptive_pipeline.py, names of main log files
  PATTERNS=('run_full_experiment.sh' 'run_adaptive_pipeline.py' 'run_full' 'adaptive_pipeline' 'optuna')
  FOUND_PIDS=()
  for p in "${PATTERNS[@]}"; do
    while read -r pid; do
      if [[ -n "$pid" ]]; then
        FOUND_PIDS+=("$pid")
      fi
    done < <(pgrep -f "$p" || true)
  done
  # Deduplicate
  if [ ${#FOUND_PIDS[@]} -gt 0 ]; then
    printf "%-8s %-6s %-6s %-8s %s\n" "PID" "%CPU" "%MEM" "ETIME" "CMD"
    for pid in $(printf "%s\n" "${FOUND_PIDS[@]}" | sort -u); do
      if ps -p "$pid" >/dev/null 2>&1; then
        ps -p "$pid" -o pid,%cpu,%mem,etime,cmd --no-headers
      fi
    done
  else
    echo "No related processes found by pgrep patterns. You can pass a specific PID manually to ps if known."
  fi
  echo ""
fi

# Start live tail
echo "Starting live tail (press Ctrl-C to stop) -- initial lines: $TAIL_LINES"
if [ -n "$FILTER_REGEX" ]; then
  echo "Filtering lines by regex: $FILTER_REGEX"
  # Use unbuffered tail and grep with --line-buffered
  exec tail -n "$TAIL_LINES" -F "$RUN_DIR"/*.log 2>/dev/null | grep --line-buffered -E "$FILTER_REGEX"
else
  exec tail -n "$TAIL_LINES" -F "$RUN_DIR"/*.log
fi
