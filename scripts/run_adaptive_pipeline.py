#!/usr/bin/env python3
"""Adaptive pipeline: LHS -> Fine -> Optuna -> Bootstrap orchestration.

AKTUALISIERT: 2026-01-15
- Phase 1 nutzt jetzt LHS statt manuellem Coarse Grid (wissenschaftlich fundiert)
- n_lhs wird adaptiv aus Datensatz-Größe berechnet (sqrt(n_tiles))
- Ermöglicht direkte Vergleichbarkeit mit Thesis Pipeline

Usage:
    python scripts/run_adaptive_pipeline.py --yes
    python scripts/run_adaptive_pipeline.py --yes --n-lhs 50
"""
import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from src.pipeline_utils import compute_fine_search_bounds, compute_optuna_bounds, compute_bootstrap_candidates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Compute adaptive default for n_lhs based on dataset size
# Faustregel: max(27, sqrt(n_tiles)) für ausreichende Abdeckung
try:
    metadata_path = ROOT / "data" / "new_all_tiles.csv"
    if metadata_path.exists():
        n_tiles = len(pd.read_csv(metadata_path))
        n_lhs_default = max(27, int(np.sqrt(n_tiles)))
        print(f"📊 Adaptive n_lhs: {n_lhs_default} (based on {n_tiles} tiles)")
    else:
        n_lhs_default = 27
        print("⚠️  Metadata not found, using default n_lhs=27")
except Exception as e:
    n_lhs_default = 27
    print(f"⚠️  Could not compute adaptive n_lhs: {e}. Using default=27")

parser = argparse.ArgumentParser()
parser.add_argument('--yes', action='store_true', help='Non-interactive')
parser.add_argument('--n-lhs', type=int, default=n_lhs_default, 
                    help=f'Number of LHS samples for exploration (default: {n_lhs_default}, computed from sqrt(n_tiles))')
parser.add_argument('--n-trials', type=int, default=200)
parser.add_argument('--n-candidates', type=int, default=500)
parser.add_argument('--n-boot', type=int, default=200)
parser.add_argument('--fine-max-runs', type=int, default=None, help='Max runs for fine sweep (smoke testing)')
parser.add_argument('--skip-optuna', action='store_true', help='Skip Optuna stage')
parser.add_argument('--skip-bootstrap-injection', action='store_true', help='Skip Bootstrap-best config generation')
parser.add_argument('--dry-run', action='store_true', help='Print commands but do not execute heavy subprocess calls')
args = parser.parse_args()

# Helper to optionally run shell commands (support --dry-run)
def run_cmd(cmd: str):
    print(f"CMD: {cmd}")
    if not args.dry_run:
        subprocess.check_call(['bash', '-lc', cmd])

# 1) Run LHS Exploration (ersetzt alten Coarse Sweep)
print('=== Phase 1: Exploration (LHS) ===')
print(f'Running LHS with {args.n_lhs} samples (replacing old manual Coarse Grid)...')
lhs_cmd = f'PYTHONPATH=. python scripts/tune_weights_and_run.py --n-samples {args.n_lhs} --seed 42'
run_cmd(lhs_cmd)

# 2) Compute Fine Bounds from LHS results
pareto_lhs = OUT / 'tuning_weights' / 'pareto' / 'pareto_solutions.csv'
if not pareto_lhs.exists():
    raise SystemExit('LHS pareto not found; aborting adaptive pipeline')

fine_bounds = compute_fine_search_bounds(str(pareto_lhs))
print(f'Computed fine bounds from LHS: {fine_bounds}')
min_distances_arg = ','.join([str(int(x)) for x in fine_bounds])

# 3) Run Fine Sweep with adaptive bounds
print('=== Phase 2: Fine Sweep (Adaptive Bounds) ===')
fine_cmd = f'PYTHONPATH=. python scripts/run_fine_sweep.py --min-distances "{min_distances_arg}"'
if args.fine_max_runs:
    fine_cmd += f' --max-runs {args.fine_max_runs}'
run_cmd(fine_cmd)

# 4) Compute Optuna bounds
pareto_fine = OUT / 'fine_sweep' / 'pareto_solutions.csv'
if not pareto_fine.exists():
    raise SystemExit('Fine pareto not found; aborting adaptive pipeline')

opt_lo, opt_hi = compute_optuna_bounds(str(pareto_fine))
center = (opt_lo + opt_hi) // 2
print(f'Optuna bounds: {opt_lo}-{opt_hi}, running Optuna with default min_distance={center}')

# 5) Run Optuna (uses our default trial bounds internally) if installed
if args.skip_optuna:
    print('Skipping Optuna stage (--skip-optuna flag provided)')
else:
    try:
        import importlib
        if importlib.util.find_spec('optuna') is None:
            print('Optuna not installed: skipping Optuna stage. Install optuna to enable.')
        else:
            print('=== Phase 3: Optimization (Optuna) ===')
            run_cmd(f'PYTHONPATH=. python scripts/optuna_optimize.py --n-trials {args.n_trials} --n-candidates {args.n_candidates} --min-distance-km {center}')
            # 6) Analyze Optuna convergence
            run_cmd('python scripts/analyze_optuna_convergence.py outputs/optuna_comparison --output-dir outputs')
    except Exception as e:
        print(f'Warning while running Optuna or analysis: {e}\nProceeding to Bootstrap stage.')

# 7) Bootstrap (on fine pareto / or optuna results as chosen)
print('=== Phase 4: Validation (Bootstrap) ===')
bootstrap_out = OUT / 'bootstrap_results.csv'
bootstrap_summary = bootstrap_out.with_name(bootstrap_out.stem + '_summary.csv')
run_cmd(f'PYTHONPATH=. python scripts/bootstrap_pareto_candidates.py --pareto {pareto_fine} --n-boot {args.n_boot} --out {bootstrap_out} --seed 42')

# 8) Apply Bootstrap Best (optional)
if not args.skip_bootstrap_injection and bootstrap_summary.exists():
    print('=== Applying Bootstrap Best ===')
    try:
        run_cmd(f'PYTHONPATH=. python scripts/apply_bootstrap_best.py --bootstrap-summary {bootstrap_summary} --write-config outputs/pipeline_config.bootstrap.yaml')
        print(f'✓ Bootstrap-best config written to outputs/pipeline_config.bootstrap.yaml')
    except Exception as e:
        print(f'Warning: Bootstrap-best application failed: {e}')
else:
    if args.skip_bootstrap_injection:
        print('Skipping Bootstrap-best injection (--skip-bootstrap-injection flag)')
    else:
        print(f'Warning: Bootstrap summary not found at {bootstrap_summary}')

print('\n' + '='*80)
print('✅ ADAPTIVE PIPELINE COMPLETE')
print('='*80)
print('Pipeline stages:')
print(f'  1. LHS Exploration: {args.n_lhs} samples')
print(f'  2. Fine Sweep: {len(fine_bounds)} adaptive bounds')
print(f'  3. Optuna: {args.n_trials} trials (center={center}km)')
print(f'  4. Bootstrap: {args.n_boot} resamples')
print('='*80)

# 9) Optional: Generate an experiment report for this adaptive run in outputs/experiments
try:
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    report_dir = OUT / 'experiments' / f'run_adaptive_{ts}'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'run_info.txt').write_text(f'Adaptive pipeline run completed: n_lhs={args.n_lhs} at {ts}\n')
    print(f'Generating experiment report in: {report_dir}')
    run_cmd(f'PYTHONPATH=. python scripts/generate_experiment_report.py --outdir "{report_dir}"')
    print(f'Report written: {report_dir / "experiment_report.md"}')
except Exception as e:
    print(f'Warning: automatic report generation failed: {e}')
