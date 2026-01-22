#!/usr/bin/env python3
"""
Run sampler comparison (QMC vs TPE vs CMA-ES) on KDR100 best selection.

This validates that sampler performance generalizes beyond Hamburg.

Usage:
    python scripts/compare_samplers_on_kdr100.py --selection-json outputs/kdr100_best_selection_info.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_sampler_on_selection(
    sampler_name: str,
    selection_tiles: list,
    n_trials: int = 500,
    tag: str = "kdr100_comparison",
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
        "--n-samples=34",
        f"--output-dir={run_dir}",
    ]

    # If specific tiles provided, add to preselection
    if selection_tiles:
        tiles_str = ",".join(map(str, selection_tiles))
        cmd.append(f"--tile-indices={tiles_str}")

    print(f"[INFO] Running {sampler_name.upper()} sampler ({n_trials} trials)...")
    print(f"       Command: {' '.join(cmd)}")

    try:
        # Run the command; we don't need the subprocess result object here
        subprocess.run(cmd, check=True, capture_output=False)

        # Read best trial
        best_trial_path = Path(run_dir) / "results" / "best_trial.json"
        if best_trial_path.exists():
            with open(best_trial_path) as f:
                best_trial = json.load(f)

            fitness = best_trial.get("fitness", best_trial.get("value"))
            print(f"[✓] {sampler_name.upper()}: Best fitness = {fitness:.4f}")

            return {
                "sampler": sampler_name,
                "n_trials": n_trials,
                "best_fitness": fitness,
                "run_dir": run_dir,
                "status": "completed",
            }
        else:
            print(f"[WARNING] No best_trial.json found for {sampler_name}")
            return {
                "sampler": sampler_name,
                "status": "failed",
                "error": "No best_trial.json found",
            }

    except Exception as e:
        print(f"[ERROR] {sampler_name} failed: {e}")
        return {
            "sampler": sampler_name,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare samplers on KDR100 selection")
    parser.add_argument(
        "--selection-json", required=True, help="Path to KDR100 selection JSON"
    )
    parser.add_argument("--n-trials", type=int, default=500, help="Trials per sampler")
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["qmc", "tpe", "cmaes"],
        help="Samplers to compare",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run samplers sequentially (default: parallel)",
    )
    parser.add_argument(
        "--output",
        default="outputs/kdr100_sampler_comparison_results.json",
        help="Results JSON",
    )

    args = parser.parse_args()

    # Load selection
    selection_path = Path(args.selection_json)
    if not selection_path.exists():
        print(f"[ERROR] Selection file not found: {selection_path}")
        return 1

    with open(selection_path) as f:
        selection_info = json.load(f)

    selected_tiles = selection_info.get("selected_tiles", [])
    print(f"[INFO] Loaded KDR100 selection: {len(selected_tiles)} tiles")
    print(f"       From run: {selection_info.get('run_id')}")
    print(f"       Best fitness: {selection_info.get('fitness'):.4f}")

    results = []

    if args.sequential:
        # Sequential execution
        for sampler in args.samplers:
            result = run_sampler_on_selection(
                sampler, selected_tiles, n_trials=args.n_trials, tag="kdr100_seq"
            )
            results.append(result)
    else:
        # Parallel execution (via subprocess backgrounding)
        # NOTE: placeholder - currently executed sequentially for determinism
        for sampler in args.samplers:
            result = run_sampler_on_selection(
                sampler, selected_tiles, n_trials=args.n_trials, tag="kdr100_par"
            )
            results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "kdr100_selection": selection_info,
        "sampler_results": results,
        "configuration": {
            "n_trials": args.n_trials,
            "sequential": args.sequential,
        },
    }

    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n[✓] Results saved: {output_path}")

    # Print summary
    print("\n=== SAMPLER COMPARISON SUMMARY (KDR100) ===")
    for result in results:
        if result["status"] == "completed":
            print(
                f"{result['sampler'].upper():8} | Fitness: {result['best_fitness']:.4f} | Run: {result['run_dir']}"
            )
        else:
            print(f"{result['sampler'].upper():8} | FAILED: {result.get('error')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
