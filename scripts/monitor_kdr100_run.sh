#!/usr/bin/env bash
# Monitor KDR100 sample run progress
# Usage: ./scripts/monitor_kdr100_run.sh

set -euo pipefail

RUN_LOG="outputs/runs/run_adaptive_20260116T214718Z.session.log"
RESULTS_DIR="outputs/runs/20260116_T214724_adaptive_full/results"

if [[ ! -f "$RUN_LOG" ]]; then
    echo "[ERROR] Log file not found: $RUN_LOG"
    exit 1
fi

echo "=== KDR100 Full Sample Run Monitor ==="
echo "Log: $RUN_LOG"
echo ""

# Check if run is still executing
if pgrep -f "run_adaptive_pipeline.py" > /dev/null; then
    echo "✓ Run is ACTIVE"
else
    echo "✓ Run appears to have FINISHED or is in idle state"
fi

echo ""
echo "=== Recent Log Output ==="
tail -20 "$RUN_LOG"

echo ""
echo "=== Results Directory Status ==="
if [[ -d "$RESULTS_DIR" ]]; then
    echo "Results dir exists: $RESULTS_DIR"
    ls -lh "$RESULTS_DIR"
    
    if [[ -f "$RESULTS_DIR/trials.csv" ]]; then
        trials=$(( $(wc -l < "$RESULTS_DIR/trials.csv") - 1 ))
        echo ""
        echo "Trials completed: $trials / 300"
    fi
    
    if [[ -f "$RESULTS_DIR/best_trial.json" ]]; then
        echo ""
        echo "Best trial found:"
        cat "$RESULTS_DIR/best_trial.json" | python3 -m json.tool | head -20
    fi
else
    echo "Results dir not yet created"
fi

echo ""
echo "=== Next Steps (when run completes) ==="
echo "1. Extract selection:       python scripts/extract_kdr100_selection.py --run-id 20260116_T214724_adaptive_full"
echo "2. Compare samplers:        python scripts/compare_samplers_on_kdr100.py --selection-json outputs/kdr100_best_selection_info.json"
echo "3. Analyze results:         python scripts/analyze_kdr100_comparison.py --results outputs/kdr100_sampler_comparison_results.json"
