#!/usr/bin/env python3
"""
Orchestrator for thesis-grade sampler evaluation.

1) Run multi-seed sampler comparisons on 'hamburg' and 'kdr100'
2) Compute best sampler per dataset and overall
3) Launch full adaptive runs (n_trials=2000) with best sampler on Hamburg and KDR100

Usage:
    python scripts/run_thesis_sampler_suite.py --seeds 42 43 44 45 46 --n-trials 500 --n-trials-full 2000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "outputs" / "runs"


def run_cmd(cmd, cwd=None):
    """Run a command inside the canonical environment wrapper if available.

    This ensures subprocesses execute inside the `dataselector` conda env via
    `scripts/exec_in_env.sh --env dataselector -- <cmd>` when the wrapper exists.
    """
    wrapper = ROOT / "scripts" / "exec_in_env.sh"
    if wrapper.exists():
        wrapped_cmd = f"{wrapper} --env dataselector -- {cmd}"
        print(f"RUN (via wrapper): {wrapped_cmd}")
        proc = subprocess.run(wrapped_cmd, shell=True, cwd=cwd)
    else:
        print(f"RUN: {cmd}")
        proc = subprocess.run(cmd, shell=True, cwd=cwd)

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def choose_best_sampler(results_dir: Path):
    # Try to read per-dataset summary files first
    summaries = []
    for dataset_dir in (results_dir).glob("*/"):
        summary_csv = dataset_dir / "summary.csv"
        if summary_csv.exists():
            summaries.append(summary_csv)

    # Fallback: some analysis scripts write a global summary.csv at results_dir
    if not summaries:
        global_summary = results_dir / "summary.csv"
        if global_summary.exists():
            summaries = [global_summary]

    if not summaries:
        raise RuntimeError(f"No summary files found in {results_dir} subfolders or {results_dir}/summary.csv")

    df_all = []
    for s in summaries:
        try:
            df = pd.read_csv(s)
            # If this is a global summary, dataset column may be missing; add dataset label from parent dir when available
            dataset = s.parent.name if s.parent != results_dir else s.parent.name
            df["dataset"] = dataset
            df_all.append(df)
        except Exception as e:
            print(f"Warning: could not read summary {s}: {e}")

    if not df_all:
        raise RuntimeError("No summary files could be read")

    df_all = pd.concat(df_all, ignore_index=True)
    # Compute mean best value per sampler across datasets
    grp = df_all.groupby("sampler")["mean"].mean().reset_index()
    grp = grp.sort_values("mean", ascending=False)
    best = grp.iloc[0]["sampler"]
    return best, grp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51], help="Random seeds for reproducibility (default: 10 seeds for thesis-grade validation)")
    parser.add_argument(
        "--n-trials", type=int, default=1000, help="Trials per sampler in comparison (default: 1000 per convergence analysis: 99% optimum at ~650 trials; 1000 provides thesis-grade robustness)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hamburg", "kdr100"],
        help="Datasets to compare on (default: hamburg + kdr100 for representative comparison)",
    )
    parser.add_argument("--samplers", nargs="+", default=["qmc", "tpe", "cmaes"], help="Samplers to compare (default: QMC, TPE, CMA-ES)")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument(
        "--n-trials-full", type=int, default=2000, help="Trials for full adaptive runs"
    )
    parser.add_argument("--n-candidates", type=int, default=None)  # Read dynamically from CSV if not set
    parser.add_argument(
        "--autoscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optuna_autoscale.py to determine best n_samples before running sampler suite",
    )
    args = parser.parse_args()

    # Dynamically read n_candidates from CSV if not set
    if args.n_candidates is None:
        csv_path = ROOT / "data" / "new_all_tiles.csv"
        if csv_path.exists():
            n_candidates = len(pd.read_csv(csv_path))
            print(f"Dynamically determined n_candidates={n_candidates} from {csv_path}")
        else:
            print(f"WARNING: {csv_path} not found, using default 676")
            n_candidates = 676
    else:
        n_candidates = args.n_candidates

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    suite_dir = OUT_BASE / f"sampler_thesis_suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # 1) Optionally run autoscale to determine n_samples and hyperparams
    best_n_samples = None
    constrain_bounds = {}
    wrapper = ROOT / "scripts" / "exec_in_env.sh"

    if args.autoscale:
        print("Running autoscale to determine best n_samples and hyperparams (this may take some time)...")
        # Prefer running autoscale inside the canonical environment if wrapper exists
        if wrapper.exists():
            autoscale_cmd = f"{wrapper} --env dataselector -- python scripts/optuna_autoscale.py --n-candidates {n_candidates}"
        else:
            autoscale_cmd = f"python scripts/optuna_autoscale.py --n-candidates {n_candidates}"
        print(autoscale_cmd)
        proc = subprocess.run(autoscale_cmd, shell=True, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            print("Warning: autoscale failed; proceeding without it")
        else:
            # Read full best JSON to extract hyperparams
            best_json = Path("outputs") / "optuna_autoscale_best_latest.json"
            try:
                if best_json.exists():
                    import json as json_mod
                    data = json_mod.loads(best_json.read_text())
                    ua = data.get("user_attrs", {})
                    
                    # Extract n_samples
                    best_n_samples = int(ua.get("n_samples", 38))
                    print(f"Autoscale selected n_samples={best_n_samples}")
                    
                    # Extract hyperparams and create constrained bounds
                    alpha = ua.get("alpha", 0.33)
                    beta = ua.get("beta", 0.40)
                    gamma = ua.get("gamma", 0.27)
                    min_dist = ua.get("min_distance_km", 28)
                    
                    # Create bounds with ±0.15 margin (constrained search)
                    margin_ab = 0.15
                    margin_md = 10
                    
                    constrain_bounds = {
                        "a_min": max(0.01, alpha - margin_ab),
                        "a_max": min(1.0, alpha + margin_ab),
                        "b_min": max(0.01, beta - margin_ab),
                        "b_max": min(1.0, beta + margin_ab),
                        "c_min": max(0.01, gamma - margin_ab),
                        "c_max": min(1.0, gamma + margin_ab),
                        "min_dist_min": max(0, int(min_dist - margin_md)),
                        "min_dist_max": int(min_dist + margin_md),
                    }
                    
                    print(f"Constrained bounds: a=[{constrain_bounds['a_min']:.3f}, {constrain_bounds['a_max']:.3f}], "
                          f"b=[{constrain_bounds['b_min']:.3f}, {constrain_bounds['b_max']:.3f}], "
                          f"c=[{constrain_bounds['c_min']:.3f}, {constrain_bounds['c_max']:.3f}], "
                          f"min_dist=[{constrain_bounds['min_dist_min']}, {constrain_bounds['min_dist_max']}]")
                else:
                    # try to read simple n_samples text file
                    sel_file = Path("outputs") / "optuna_autoscale_selected_n_samples.txt"
                    if sel_file.exists():
                        best_n_samples = int(sel_file.read_text().strip())
                        print(f"Autoscale selected n_samples (from text file)={best_n_samples}")
            except Exception as e:
                print(f"Warning: could not parse autoscale output: {e}")

    # 1b) Run compare_samplers_multi_seed.py
    out_arg = f"--output {suite_dir}"
    seeds_arg = " ".join(str(s) for s in args.seeds)
    samplers_arg = " ".join(args.samplers)
    datasets_arg = " ".join(args.datasets)

    compare_cmd = f"python scripts/compare_samplers_multi_seed.py --samplers {samplers_arg} --seeds {seeds_arg} --n-trials {args.n_trials} --datasets {datasets_arg} --sequential --output {suite_dir} --n-candidates {n_candidates}"
    if best_n_samples is not None:
        compare_cmd += f" --n-samples {best_n_samples}"
    
    # Add constrained bounds if available
    if constrain_bounds:
        compare_cmd += (
            f" --constrain-a-min {constrain_bounds['a_min']}"
            f" --constrain-a-max {constrain_bounds['a_max']}"
            f" --constrain-b-min {constrain_bounds['b_min']}"
            f" --constrain-b-max {constrain_bounds['b_max']}"
            f" --constrain-c-min {constrain_bounds['c_min']}"
            f" --constrain-c-max {constrain_bounds['c_max']}"
            f" --constrain-min-dist-min {constrain_bounds['min_dist_min']}"
            f" --constrain-min-dist-max {constrain_bounds['min_dist_max']}"
        )

    run_cmd(compare_cmd)

    # 2) Choose best sampler
    try:
        best, table = choose_best_sampler(suite_dir)
        print(f"Best sampler (overall mean of dataset summaries): {best}")
        (suite_dir / "best_sampler_summary.json").write_text(
            json.dumps(
                {"best": best, "summary_table": table.to_dict(orient="records")},
                indent=2,
            )
        )
    except Exception as e:
        print(f"ERROR selecting best sampler: {e}")
        sys.exit(1)

    # 3) Launch full adaptive runs with best sampler: Hamburg and KDR100 (no --hamburg == full)
    # Use exec_in_env.sh wrapper if available
    wrapper = ROOT / "scripts" / "exec_in_env.sh"

    # If the full adaptive script is missing, skip these heavy runs and log clearly
    adaptive_script = ROOT / "scripts" / "run_adaptive_pipeline.py"
    if not adaptive_script.exists():
        print(f"WARNING: {adaptive_script} not found - skipping full adaptive runs. Create this script to run full XXL jobs.")
    else:
        # Hamburg full run
        run_name_h = f"suite_full_{best}_hamburg_{timestamp}"
        cmd_h = f"env PYTHONPATH=. python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {n_candidates} --sampler {best} --seed {args.seeds[0]} --hamburg"
        print(f"Launching full Hamburg run: {cmd_h}")
        run_cmd(cmd_h)

        # KDR100 full run (no preselection)
        run_name_k = f"suite_full_{best}_kdr100_{timestamp}"
        cmd_k = f"env PYTHONPATH=. python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {n_candidates} --sampler {best} --seed {args.seeds[0]}"
        print(f"Launching full KDR100 run: {cmd_k}")
        run_cmd(cmd_k)

    print("\n=== SUITE COMPLETE ===")
    print(f"Results and artifacts: {suite_dir}")
