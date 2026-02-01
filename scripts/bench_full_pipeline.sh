#!/usr/bin/env bash
set -euo pipefail

THREADS=(1 4 8)
mkdir -p outputs/bench_threads/full_pipeline

for T in "${THREADS[@]}"; do
  echo "\n=== threads=$T ==="
  export OMP_NUM_THREADS=$T
  export MKL_NUM_THREADS=$T

  rm -f outputs/optuna_results.csv outputs/bootstrap_results.csv
  LOG=outputs/bench_threads/full_pipeline/run_full_threads_${T}.log

  START=$(date +%s)
  # run full adaptive pipeline with small budgets
  python scripts/run_adaptive_pipeline.py --yes --n-lhs 10 --n-trials 20 --n-candidates 100 --n-boot 20 --fine-max-runs 5 > "$LOG" 2>&1 || true
  END=$(date +%s)
  ELAPSED=$((END-START))

  echo "threads=$T elapsed=${ELAPSED}s"
  echo "--- tail log ---"
  tail -n 60 "$LOG" || true

  echo "--- optuna head ---"
  if [ -f outputs/optuna_results.csv ]; then head -n 8 outputs/optuna_results.csv; else echo "no optuna_results"; fi

  echo "--- bootstrap head ---"
  if [ -f outputs/bootstrap_results.csv ]; then head -n 8 outputs/bootstrap_results.csv; else echo "no bootstrap_results"; fi

  cp -v outputs/optuna_results.csv outputs/bench_threads/full_pipeline/optuna_threads_${T}.csv 2>/dev/null || true
  cp -v outputs/bootstrap_results.csv outputs/bench_threads/full_pipeline/bootstrap_threads_${T}.csv 2>/dev/null || true

done

echo; echo "Summary artifacts in outputs/bench_threads/full_pipeline/"; ls -lh outputs/bench_threads/full_pipeline || true
