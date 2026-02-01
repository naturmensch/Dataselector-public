#!/usr/bin/env python3
<<<<<<< HEAD
"""Compare Samplers on KDR100.

This script compares different sampling methods on KDR100 dataset
by evaluating their performance on selection tasks.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

ROOT = Path(__file__).resolve().parents[1]


def load_selection_results(selection_json: str) -> Dict:
    """Load selection results from JSON."""
    with open(selection_json, 'r') as f:
        return json.load(f)


def compare_samplers(
    selection_json: str,
    output_dir: str = 'outputs'
):
    """Compare different samplers based on selection results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = load_selection_results(selection_json)
    print(f"Loaded selection results with {len(data)} entries")

    # Extract sampler information
    samplers = {}
    for key, value in data.items():
        if isinstance(value, dict) and 'sampler' in value:
            sampler = value['sampler']
            if sampler not in samplers:
                samplers[sampler] = []
            samplers[sampler].append(value)

    if not samplers:
        print("No sampler data found in JSON")
        return

    print(f"Found {len(samplers)} different samplers: {list(samplers.keys())}")

    # Compare metrics
    metrics_to_compare = ['temporal_std', 'wwi_percent', 'jaccard_with_original',
                         'selection_size', 'compute_time']

    results = []
    for sampler, runs in samplers.items():
        for run in runs:
            result = {'sampler': sampler}
            for metric in metrics_to_compare:
                if metric in run:
                    result[metric] = run[metric]
                else:
                    result[metric] = None
            results.append(result)

    results_df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / 'sampler_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved comparison results to {csv_path}")

    # Plot comparisons
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_compare):
        if i >= 6:
            break

        ax = axes[i]
        valid_data = results_df.dropna(subset=[metric])

        if valid_data.empty:
            ax.text(0.5, 0.5, f'No data for {metric}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Box plot
        sampler_groups = [valid_data[valid_data['sampler'] == s][metric].values
                         for s in valid_data['sampler'].unique()]

        if sampler_groups:
            ax.boxplot(sampler_groups, labels=valid_data['sampler'].unique())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for i in range(len(metrics_to_compare), 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plot_path = output_dir / 'sampler_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plots to {plot_path}")

    # Print summary statistics
    print("\nSampler Comparison Summary:")
    summary = results_df.groupby('sampler').agg({
        'temporal_std': ['mean', 'std', 'count'],
        'wwi_percent': ['mean', 'std'],
        'jaccard_with_original': ['mean', 'std'],
        'selection_size': ['mean', 'std'],
        'compute_time': ['mean', 'std']
    }).round(4)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='Compare Samplers on KDR100')
    parser.add_argument('--selection-json', required=True,
                       help='Path to selection results JSON file')

    args = parser.parse_args()

    compare_samplers(args.selection_json)


if __name__ == '__main__':
    main()
=======
"""
Run sampler comparison (QMC vs TPE vs CMA-ES) on KDR100 best selection.

This validates that sampler performance generalizes beyond Hamburg.

Usage:
    python scripts/compare_samplers_on_kdr100.py --selection-json outputs/kdr100_best_selection_info.json
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_sampler_on_selection(
    sampler_name: str, 
    selection_tiles: list, 
    n_trials: int = 500, 
    tag: str = "kdr100_comparison"
) -> dict:
    """Run a single sampler on the given tile selection."""
    
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = f"outputs/runs/{timestamp}_{sampler_name}_{tag}_500trials"
    
    # Prepare Optuna command
    cmd = [
        "python3",
        "scripts/optuna_optimize.py",
        f"--sampler={sampler_name}",
        f"--n-trials={n_trials}",
        f"--n-samples=34",
        f"--output-dir={run_dir}",
    ]
    
    # If specific tiles provided, add to preselection
    if selection_tiles:
        tiles_str = ",".join(map(str, selection_tiles))
        cmd.append(f"--tile-indices={tiles_str}")
    
    print(f"[INFO] Running {sampler_name.upper()} sampler ({n_trials} trials)...")
    print(f"       Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # Read best trial
        best_trial_path = Path(run_dir) / "results" / "best_trial.json"
        if best_trial_path.exists():
            with open(best_trial_path) as f:
                best_trial = json.load(f)
            
            fitness = best_trial.get('fitness', best_trial.get('value'))
            print(f"[✓] {sampler_name.upper()}: Best fitness = {fitness:.4f}")
            
            return {
                'sampler': sampler_name,
                'n_trials': n_trials,
                'best_fitness': fitness,
                'run_dir': run_dir,
                'status': 'completed',
            }
        else:
            print(f"[WARNING] No best_trial.json found for {sampler_name}")
            return {
                'sampler': sampler_name,
                'status': 'failed',
                'error': 'No best_trial.json found',
            }
    
    except Exception as e:
        print(f"[ERROR] {sampler_name} failed: {e}")
        return {
            'sampler': sampler_name,
            'status': 'failed',
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare samplers on KDR100 selection")
    parser.add_argument("--selection-json", required=True, help="Path to KDR100 selection JSON")
    parser.add_argument("--n-trials", type=int, default=500, help="Trials per sampler")
    parser.add_argument("--samplers", nargs="+", default=["qmc", "tpe", "cmaes"], help="Samplers to compare")
    parser.add_argument("--sequential", action="store_true", help="Run samplers sequentially (default: parallel)")
    parser.add_argument("--output", default="outputs/kdr100_sampler_comparison_results.json", help="Results JSON")
    
    args = parser.parse_args()
    
    # Load selection
    selection_path = Path(args.selection_json)
    if not selection_path.exists():
        print(f"[ERROR] Selection file not found: {selection_path}")
        return 1
    
    with open(selection_path) as f:
        selection_info = json.load(f)
    
    selected_tiles = selection_info.get('selected_tiles', [])
    print(f"[INFO] Loaded KDR100 selection: {len(selected_tiles)} tiles")
    print(f"       From run: {selection_info.get('run_id')}")
    print(f"       Best fitness: {selection_info.get('fitness'):.4f}")
    
    results = []
    
    if args.sequential:
        # Sequential execution
        for sampler in args.samplers:
            result = run_sampler_on_selection(
                sampler,
                selected_tiles,
                n_trials=args.n_trials,
                tag="kdr100_seq"
            )
            results.append(result)
    else:
        # Parallel execution (via subprocess backgrounding)
        import time
        processes = {}
        
        for sampler in args.samplers:
            # TODO: Implement true parallel execution with multiprocessing or Celery
            # For now, run sequentially
            result = run_sampler_on_selection(
                sampler,
                selected_tiles,
                n_trials=args.n_trials,
                tag="kdr100_par"
            )
            results.append(result)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'kdr100_selection': selection_info,
        'sampler_results': results,
        'configuration': {
            'n_trials': args.n_trials,
            'sequential': args.sequential,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n[✓] Results saved: {output_path}")
    
    # Print summary
    print("\n=== SAMPLER COMPARISON SUMMARY (KDR100) ===")
    for result in results:
        if result['status'] == 'completed':
            print(f"{result['sampler'].upper():8} | Fitness: {result['best_fitness']:.4f} | Run: {result['run_dir']}")
        else:
            print(f"{result['sampler'].upper():8} | FAILED: {result.get('error')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
>>>>>>> ci/add-smoke-tests
