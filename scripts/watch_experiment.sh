#!/usr/bin/env bash
# Simple watcher to tail all .log files for a given run directory
# Usage: ./scripts/watch_experiment.sh outputs/experiments/run_20260115T143500Z

if [ -z "$1" ]; then
  echo "Usage: $0 <run_dir>"
  exit 1
fi
RUN_DIR="$1"
if [ ! -d "$RUN_DIR" ]; then
  echo "Run dir not found: $RUN_DIR"
  exit 1
fi

LOGS=("$RUN_DIR"/*.log)
if [ ${#LOGS[@]} -eq 0 ]; then
  echo "No log files found in $RUN_DIR"
  exit 0
fi

# Use tail -f to follow logs; combine them with prefixed filename
for f in "$RUN_DIR"/*.log; do
  echo "==> $f <=="
done

echo "Starting live tail of all logs. Press Ctrl-C to stop."
exec tail -n 200 -f "$RUN_DIR"/*.log
