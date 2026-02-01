#!/bin/bash
# Quick status check for running experiments

echo "═════════════════════════════════════════════════════════════"
echo "EXPERIMENT STATUS CHECK"
echo "═════════════════════════════════════════════════════════════"
echo ""

# Check running processes
echo "📊 Running Processes:"
ps aux | grep -E "(compare_samplers|bootstrap_final|optuna_optimize)" | grep -v grep | while read line; do
    echo "  • $line"
done
echo ""

# Check sampler comparison
if [ -f "outputs/sampler_comparison.log" ]; then
    echo "🔬 Sampler Comparison:"
    tail -5 outputs/sampler_comparison.log | sed 's/^/  /'
    echo ""
fi

# Check bootstrap
if [ -f "outputs/bootstrap_hamburg.log" ]; then
    echo "📈 Bootstrap UQ (Hamburg):"
    tail -2 outputs/bootstrap_hamburg.log | sed 's/^/  /'
    echo ""
fi

# Count completed runs
echo "✅ Completed Runs:"
if [ -d "outputs/runs" ]; then
    completed=$(find outputs/runs -name "manifest.json" -exec grep -l '"status": "complete"' {} \; | wc -l)
    running=$(find outputs/runs -name "manifest.json" -exec grep -l '"status": "running"' {} \; | wc -l)
    echo "  Completed: $completed"
    echo "  Running:   $running"
    echo ""
fi

# Latest runs
echo "🕐 Latest Runs (top 5):"
ls -td outputs/runs/* 2>/dev/null | head -5 | while read dir; do
    name=$(basename "$dir")
    if [ -f "$dir/manifest.json" ]; then
        status=$(grep -o '"status": "[^"]*"' "$dir/manifest.json" | cut -d'"' -f4)
        echo "  • $name [$status]"
    else
        echo "  • $name [no manifest]"
    fi
done
echo ""

echo "═════════════════════════════════════════════════════════════"
echo "Monitor logs:"
echo "  sampler:   tail -f outputs/sampler_comparison.log"
echo "  bootstrap: tail -f outputs/bootstrap_hamburg.log"
echo "═════════════════════════════════════════════════════════════"
