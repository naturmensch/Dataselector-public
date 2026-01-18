#!/usr/bin/env python3
"""Adaptive pipeline: exploration (Sobol default, LHS optional) -> Fine -> Optuna -> Bootstrap orchestration.

AKTUALISIERT: 2026-01-15
- Phase 1 historisch mit LHS eingeführt; aktuell ist Sobol (QMC) Default, LHS bleibt als Option via --sampler
- n_lhs wird adaptiv aus Datensatz-Größe berechnet (sqrt(n_tiles))
- Ermöglicht direkte Vergleichbarkeit mit Thesis Pipeline

Usage:
    python scripts/run_adaptive_pipeline.py --yes
    python scripts/run_adaptive_pipeline.py --yes --n-lhs 50
"""
import subprocess
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from src.pipeline_utils import compute_fine_search_bounds, compute_optuna_bounds, compute_bootstrap_candidates
import shlex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Read dataset size (optional) - used by legacy heuristic
try:
    metadata_path = ROOT / "data" / "new_all_tiles.csv"
    if metadata_path.exists():
        n_tiles = len(pd.read_csv(metadata_path))
        print(f"📊 Found {n_tiles} data tiles (used by legacy heuristics)")
    else:
        n_tiles = None
        print("⚠️  Metadata not found; legacy heuristics will use defaults")
except Exception as e:
    n_tiles = None
    print(f"⚠️  Could not read metadata: {e}; legacy heuristics will use defaults")

parser = argparse.ArgumentParser()
parser.add_argument('--yes', action='store_true', help='Non-interactive')
parser.add_argument('--n-lhs', type=int, default=None, 
                    help='Number of samples for exploration (if omitted, computed via --n-initial-strategy)')
parser.add_argument('--n-initial-strategy', choices=['legacy','modern'], default='modern',
                    help='Strategy to compute initial sample size: legacy=max(27,sqrt(n_tiles)) or modern=2*D^2 (default: modern)')
parser.add_argument('--n-dimensions', type=int, default=3, help='Number of optimization dimensions (default: 3 weights)')
parser.add_argument('--sampler', choices=['lhs','sobol'], default='sobol',
                    help='Sampler for initial exploration (default: sobol)')
parser.add_argument('--optuna-sampler', choices=['tpe','qmc','cmaes'], default='tpe',
                    help='Sampler to use within Optuna optimization (default: tpe)')
parser.add_argument('--n-trials', type=int, default=200)
parser.add_argument('--n-candidates', type=int, default=500)
parser.add_argument('--n-boot', type=int, default=200)
parser.add_argument('--fine-max-runs', type=int, default=None, help='Max runs for fine sweep (smoke testing)')
parser.add_argument('--skip-optuna', action='store_true', help='Skip Optuna stage')
parser.add_argument('--skip-bootstrap-injection', action='store_true', help='Skip Bootstrap-best config generation')
parser.add_argument('--skip-exploration', action='store_true', help='Skip Exploration (Sobol/LHS) phase')
parser.add_argument('--skip-fine', action='store_true', help='Skip Fine Sweep phase')
parser.add_argument('--dry-run', action='store_true', help='Print commands but do not execute heavy subprocess calls')
# Pre-selection / seeding options
parser.add_argument('--pre-names', type=str, nargs='*', default=None, help='Optional pre-selected tile names (e.g. Hamburg)')
parser.add_argument('--pre-indices', type=int, nargs='*', default=None, help='Optional pre-selected tile indices')
parser.add_argument('--hamburg', action='store_true', help='Convenience flag: pre-select Hamburg (name match)')
parser.add_argument('--KDR146', action='store_true', help='Convenience flag: pre-select tile KDR_146 (index or name)')
# Experiment / reproducibility options
parser.add_argument('--exp-name', type=str, default='adaptive_full', help='Experiment name for ExperimentManager')
parser.add_argument('--exp-desc', type=str, default='', help='Experiment description')
parser.add_argument('--seed', type=int, default=42, help='Global seed to pass to all stages')
args = parser.parse_args()
# Enforce single-threaded numerical libs for reproducibility unless user overrides
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Initialize ExperimentManager for the entire adaptive run and export its run_dir to sub-stages
import os
from src.experiment_manager import ExperimentManager
em = ExperimentManager(name=args.exp_name, description=args.exp_desc, metadata={'seed': args.seed})
em.save_config('run', {'n_lhs': args.n_lhs, 'n_trials': args.n_trials, 'n_candidates': args.n_candidates, 'seed': args.seed})
em.save_manifest()
# expose run dir to sub-scripts via ENV
os.environ['EXPERIMENT_RUN_DIR'] = str(em.run_dir)
print(em.summary())

# If user didn't provide n_lhs explicitly, compute using chosen strategy
from src.pipeline_utils import compute_adaptive_n_initial
def _next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p

if args.n_lhs is None:
    args.n_lhs = compute_adaptive_n_initial(args.n_dimensions, n_tiles=n_tiles, strategy=args.n_initial_strategy)
    print(f"📊 Adaptive n_lhs: {args.n_lhs} (strategy={args.n_initial_strategy})")
else:
    print(f"📊 Using user-specified n_lhs: {args.n_lhs}")

# If using Sobol QMC sampler, prefer a power-of-two sample size (better balancing properties)
if args.sampler == 'sobol':
    adjusted = _next_power_of_two(args.n_lhs)
    if adjusted != args.n_lhs:
        print(f"⚠ Using Sobol sampler: rounding n_lhs {args.n_lhs} -> next power of two {adjusted}")
        args.n_lhs = adjusted

# Helper to optionally run shell commands (support --dry-run)
def run_cmd(cmd: str):
    print(f"CMD: {cmd}")
    if not args.dry_run:
        # Use exec_in_env wrapper so the same command runs inside the canonical conda env if present
        wrapper = Path(__file__).resolve().parents[1] / 'scripts' / 'exec_in_env.sh'
        if wrapper.exists():
            subprocess.check_call([str(wrapper), '--env', os.environ.get('ENV_NAME','dataselector'), '--', 'bash', '-lc', cmd])
        else:
            subprocess.check_call(['bash', '-lc', cmd])

# 1) Run Exploration (ersetzt alten Coarse Sweep)
if args.skip_exploration:
    print('=== Phase 1: Exploration SKIPPED (--skip-exploration) ===')
    pre_names = list(args.pre_names) if args.pre_names is not None else []
    pre_indices = list(args.pre_indices) if args.pre_indices is not None else []
    pre_arg = ''
    if pre_names:
        names_quoted = ' '.join(shlex.quote(n) for n in pre_names)
        pre_arg += f" --pre-names {names_quoted}"
    if pre_indices:
        idxs = ' '.join(str(int(x)) for x in pre_indices)
        pre_arg += f" --pre-indices {idxs}"
    print(f"Using pre-selected names: {pre_names if pre_names else None}, pre-selected indices: {pre_indices if pre_indices else None}")
else:
    print(f'=== Phase 1: Exploration ({args.sampler.upper()}) ===')
    print(f'Running {args.sampler} with {args.n_lhs} samples (replacing old manual Coarse Grid)...')
    # Build preselection CLI snippet, allow convenience flags
    pre_names = list(args.pre_names) if args.pre_names is not None else []
    if args.hamburg:
        pre_names.append('Hamburg')
    if args.KDR146:
        # name used in some configs: 'KDR_146'
        pre_names.append('KDR_146')
    pre_indices = list(args.pre_indices) if args.pre_indices is not None else []
    pre_arg = ''
    if pre_names:
        names_quoted = ' '.join(shlex.quote(n) for n in pre_names)
        pre_arg += f" --pre-names {names_quoted}"
    if pre_indices:
        idxs = ' '.join(str(int(x)) for x in pre_indices)
        pre_arg += f" --pre-indices {idxs}"

    # Log preselection choices for this run
    print(f"Using pre-selected names: {pre_names if pre_names else None}, pre-selected indices: {pre_indices if pre_indices else None}")

    lhs_cmd = f'PYTHONPATH=. python scripts/tune_weights_and_run.py --n-samples {args.n_lhs} --seed {args.seed} --sampler {args.sampler}{pre_arg}'
    # Run through wrapper when available to ensure canonical env usage
    run_cmd(lhs_cmd)

# 2) Compute Fine Bounds from LHS results
pareto_lhs = OUT / 'tuning_weights' / 'pareto' / 'pareto_solutions.csv'
if args.skip_fine:
    print('=== Phase 2: Fine Sweep SKIPPED (--skip-fine) ===')
    # If skipping fine, prefer an existing fine pareto if present, else fall back to exploration pareto
    if (OUT / 'fine_sweep' / 'pareto_solutions.csv').exists():
        pareto_fine = OUT / 'fine_sweep' / 'pareto_solutions.csv'
        print(f'Using existing fine pareto: {pareto_fine}')
    elif pareto_lhs.exists():
        pareto_fine = pareto_lhs
        print(f'No fine pareto found; using exploration pareto: {pareto_fine}')
    else:
        raise SystemExit('No pareto available to proceed after skipping fine sweep; aborting')
    # compute bounds from available pareto
    fine_bounds = compute_fine_search_bounds(str(pareto_fine))
else:
    if not pareto_lhs.exists():
        raise SystemExit('Exploration pareto not found; aborting adaptive pipeline')
    fine_bounds = compute_fine_search_bounds(str(pareto_lhs))
    print(f'Computed fine bounds from exploration results: {fine_bounds}')
min_distances_arg = ','.join([str(int(x)) for x in fine_bounds])

# 3) Run Fine Sweep with adaptive bounds
if args.skip_fine:
    print('Skipping fine sweep execution (--skip-fine)')
else:
    print('=== Phase 2: Fine Sweep (Adaptive Bounds) ===')
    fine_cmd = f'PYTHONPATH=. python scripts/run_fine_sweep.py --min-distances "{min_distances_arg}"{pre_arg}'
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
        # importlib.util is a submodule that may not be directly bound as an attribute in
        # some interpreters; import it explicitly for a robust feature check
        try:
            import importlib.util as importlib_util
            has_optuna = importlib_util.find_spec('optuna') is not None
        except Exception:
            import importlib
            try:
                has_optuna = importlib.find_spec('optuna') is not None
            except Exception as e_inner:
                print(f'Warning checking for optuna using importlib: {e_inner}')
                has_optuna = False

        if not has_optuna:
            print('Optuna not found in the current environment: skipping Optuna stage. Install optuna to enable.')
        else:
            print('=== Phase 3: Optimization (Optuna) ===')
            # Run Optuna with computed n_samples bounds
            print(f'Running Optuna with n_samples range: {opt_lo}-{opt_hi} and min_distance center {center}km')
            optuna_cmd = (
                f'PYTHONPATH=. python scripts/optuna_optimize.py '
                f'--n-trials {args.n_trials} '
                f'--n-candidates {args.n_candidates} '
                f'--n-samples-min {opt_lo} '
                f'--n-samples-max {opt_hi} '
                f'--min-distance-km {center} '
                f'--sampler {args.optuna_sampler} '
                f'--seed {args.seed}{pre_arg}'
            )
            run_cmd(optuna_cmd)
            # 6) Analyze Optuna convergence
            trials_path = Path(em.run_dir) / 'results' / 'trials.csv'
            if trials_path.exists():
                run_cmd(f'python scripts/analyze_optuna_convergence.py "{trials_path}" --output-dir "{em.run_dir}/reports"')
            else:
                run_cmd(f'python scripts/analyze_optuna_convergence.py outputs/optuna_comparison --output-dir "{em.run_dir}/reports" || echo "Warning: Optuna convergence analysis failed (non-critical)"')
    except Exception as e:
        print('Warning while running Optuna or analysis. Proceeding to Bootstrap stage. Error:')
        print(repr(e))

# 7) Bootstrap (on fine pareto / or optuna results as chosen)
print('=== Phase 4: Validation (Bootstrap) ===')
bootstrap_out = Path(em.run_dir) / 'results' / 'bootstrap_results.csv'
bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
bootstrap_summary = bootstrap_out.with_name(bootstrap_out.stem + '_summary.csv')
run_cmd(f'PYTHONPATH=. python scripts/bootstrap_pareto_candidates.py --pareto {pareto_fine} --n-boot {args.n_boot} --out {bootstrap_out} --seed {args.seed}{pre_arg}')

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
print(f'  1. {args.sampler.upper()} Exploration: {args.n_lhs} samples')
print(f'  2. Fine Sweep: {len(fine_bounds)} adaptive bounds')
print(f'  3. Optuna: {args.n_trials} trials (center={center}km)')
print(f'  4. Bootstrap: {args.n_boot} resamples')
print('='*80)

# 9) Optional: Generate an experiment report for this adaptive run in outputs/experiments
try:
    report_dir = Path(em.run_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'run_info.txt').write_text(f'Adaptive pipeline run completed: sampler={args.sampler} n_lhs={args.n_lhs} at {datetime.utcnow().isoformat()}Z\n')
    print(f'Generating experiment report in: {report_dir}')
    run_cmd(f'PYTHONPATH=. python scripts/generate_experiment_report.py --outdir "{report_dir}"')
    print(f'Report written: {report_dir / "experiment_report.md"}')
except Exception as e:
    print(f'Warning: automatic report generation failed: {e}')
