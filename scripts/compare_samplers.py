#!/usr/bin/env python3
"""Scientific sampler comparison: QMC vs TPE vs CMA-ES.

Runs identical optimization problems with different samplers to empirically
validate sampler choice for thesis.

Usage:
    python scripts/compare_samplers.py --n-trials 500 --n-candidates 800 --hamburg

Generates:
    outputs/runs/sampler_comparison_<timestamp>/
        ├── qmc_500trials/
        ├── tpe_500trials/
        ├── cmaes_500trials/
        └── comparison_summary.csv
"""

import argparse
import logging
import multiprocessing
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

from typing import Optional  # noqa: E402


def run_single_sampler(
    sampler: str,
    n_trials: int,
    n_candidates: int,
    seed: int,
    preselection_flag: str,
    exp_description: str,
) -> Optional[str]:
    """Run a single sampler optimization (returns run_dir or None).

    This function is defined at module level to be picklable for multiprocessing.
    """
    import subprocess

    exp_name = f"{sampler}_{n_trials}trials"
    logger.info(f"Starting optimization for {sampler.upper()}...")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "optuna_optimize.py"),
        "--n-trials",
        str(n_trials),
        "--n-candidates",
        str(n_candidates),
        "--n-samples-min",
        "30",
        "--n-samples-max",
        "50",
        "--sampler",
        sampler,
        "--seed",
        str(seed),
        "--exp-name",
        exp_name,
        "--exp-desc",
        f"{exp_description} ({sampler})",
    ]

    if preselection_flag:
        cmd.append(preselection_flag)

    try:
        # Run optimization, capturing output to avoid interleaving
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Sampler {sampler} failed:\n{result.stderr}")
            return None

        # Find latest run dir
        run_dirs = sorted((ROOT / "outputs" / "runs").glob(f"*{exp_name}"))
        if not run_dirs:
            logger.warning(f"Could not find run directory for {sampler}")
            return None

        run_dir = run_dirs[-1]
        trials_csv = run_dir / "results" / "trials.csv"

        # Verify output integrity with a short retry loop to avoid races on shared FS
        n_retries = 3
        for attempt in range(n_retries):
            if trials_csv.exists() and trials_csv.stat().st_size >= 10:
                break
            if attempt < n_retries - 1:
                logger.info(
                    f"Waiting for trials.csv for {sampler} (attempt {attempt+1}/{n_retries})"
                )
                time.sleep(0.5)
        else:
            if not trials_csv.exists():
                logger.error(
                    f"Run {sampler} finished but {trials_csv} is missing after {n_retries} retries."
                )
            else:
                logger.error(
                    f"Run {sampler} finished but {trials_csv} seems empty/corrupt after {n_retries} retries."
                )
            return None

        return str(run_dir)

    except Exception as e:
        logger.error(f"Exception running {sampler}: {e}")
        return None


def run_sampler_comparison(
    samplers=["qmc", "tpe", "cmaes"],
    n_trials=500,
    n_candidates=673,
    seed=42,
    preselection_flag="--hamburg",
    exp_description="Sampler comparison study",
):
    """Run Optuna optimization with different samplers and compare results."""

    timestamp = datetime.now().strftime("%Y%m%d_T%H%M%S")
    base_run = ROOT / "outputs" / "runs" / f"sampler_comparison_{timestamp}"
    base_run.mkdir(parents=True, exist_ok=True)

    results = []

    # Prepare arguments for workers
    tasks = [
        (sampler, n_trials, n_candidates, seed, preselection_flag, exp_description)
        for sampler in samplers
    ]

    run_dirs_str = []

    # Execute runs (Parallel with Fallback)
    try:
        # Prefer 'fork' on Linux/Mac for speed/compatibility, default elsewhere
        ctx = (
            multiprocessing.get_context("fork")
            if "fork" in multiprocessing.get_all_start_methods()
            else multiprocessing.get_context()
        )
        try:
            # Prefer an integer number of CPUs; ctx.cpu_count() may be mocked or
            # return non-int values (e.g., MagicMock in tests), so cast defensively.
            n_cpus = int(ctx.cpu_count())
        except Exception:
            # If context does not provide cpu_count, default to conservative 1
            n_cpus = 1
        n_procs = min(len(samplers), max(1, n_cpus))
        logger.info(f"Running comparison with {n_procs} processes...")

        with ctx.Pool(processes=n_procs) as pool:
            run_dirs_str = pool.starmap(run_single_sampler, tasks)

    except (OSError, pickle.PicklingError, ValueError, Exception) as e:
        logger.warning(
            f"Parallel execution failed ({e}). Falling back to serial execution."
        )
        run_dirs_str = [run_single_sampler(*t) for t in tasks]

    # Analyze results
    for i, run_dir_s in enumerate(run_dirs_str):
        sampler = samplers[i]
        if not run_dir_s:
            continue

        run_dir = Path(run_dir_s)
        trials_csv = run_dir / "results" / "trials.csv"

        if trials_csv.exists():
            df = pd.read_csv(trials_csv)
            df = df[df["value"].notna()]

            if len(df) > 0:
                best_value = df["value"].max()
                best_trial_num = df.loc[df["value"].idxmax(), "trial_number"]
                mean_value = df["value"].mean()
                std_value = df["value"].std()

                # Convergence: trial where 99% of best is reached
                cumulative_best = df["value"].expanding().max()
                threshold = best_value * 0.99
                conv_idx = (
                    (cumulative_best >= threshold).idxmax()
                    if (cumulative_best >= threshold).any()
                    else len(df) - 1
                )

                results.append(
                    {
                        "sampler": sampler,
                        "n_trials": len(df),
                        "best_value": best_value,
                        "best_trial_number": best_trial_num,
                        "mean_value": mean_value,
                        "std_value": std_value,
                        "convergence_trial": conv_idx,
                        "convergence_ratio": conv_idx / len(df),
                        "run_dir": str(run_dir),
                    }
                )

                logger.info(
                    f"✓ {sampler.upper()} Results: Best={best_value:.4f} (Trial {best_trial_num}), Conv={conv_idx}"
                )

    if not results:
        logger.error("No results collected!")
        return 1

    # Save comparison summary
    df_results = pd.DataFrame(results)
    summary_file = base_run / "comparison_summary.csv"
    df_results.to_csv(summary_file, index=False)
    logger.info(f"Saved comparison summary: {summary_file}")

    # Create comparison plots
    create_comparison_plots(df_results, results, base_run)

    # Print final summary
    print(f"\n{'='*70}")
    print("SAMPLER COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(
        df_results[
            ["sampler", "best_value", "convergence_trial", "convergence_ratio"]
        ].to_string(index=False)
    )
    print(f"{'='*70}\n")

    # Recommendation
    best_sampler = df_results.loc[df_results["best_value"].idxmax(), "sampler"]
    fastest_conv = df_results.loc[df_results["convergence_ratio"].idxmin(), "sampler"]

    print("📊 Scientific Assessment:")
    print(f"  Best objective value: {best_sampler.upper()}")
    print(f"  Fastest convergence: {fastest_conv.upper()}")

    if best_sampler == fastest_conv:
        print(
            f"  ✅ RECOMMENDATION: Use {best_sampler.upper()} (best performance + fastest convergence)"
        )
    else:
        print(
            f"  ⚖️  TRADE-OFF: {best_sampler.upper()} for quality, {fastest_conv.upper()} for speed"
        )

    print(f"\n{'='*70}\n")

    return 0


def create_comparison_plots(df_summary, results, output_dir):
    """Create visualization plots for sampler comparison."""
    # Load trial histories
    histories = {}
    for res in results:
        run_dir = Path(res["run_dir"])
        trials_csv = run_dir / "results" / "trials.csv"
        if trials_csv.exists():
            df = pd.read_csv(trials_csv)
            df = df[df["value"].notna()]
            histories[res["sampler"]] = df

    if not histories:
        return

    # Plot 1: Convergence curves
    fig, ax = plt.subplots(figsize=(10, 6))

    for sampler, df in histories.items():
        cumulative_best = df["value"].expanding().max()
        ax.plot(cumulative_best.values, label=sampler.upper(), linewidth=2, alpha=0.8)

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(
        "Sampler Comparison: Convergence Curves", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    conv_plot = output_dir / "convergence_comparison.png"
    plt.savefig(conv_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved convergence plot: {conv_plot}")

    # Plot 2: Box plot of objective values
    fig, ax = plt.subplots(figsize=(8, 6))

    data_for_box = []
    labels_for_box = []
    for sampler, df in histories.items():
        data_for_box.append(df["value"].values)
        labels_for_box.append(sampler.upper())

    # Use tick_labels instead of labels to avoid Matplotlib warning
    bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)

    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title(
        "Sampler Comparison: Value Distribution", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    box_plot = output_dir / "distribution_comparison.png"
    plt.savefig(box_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved distribution plot: {box_plot}")

    # Plot 3: Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Best values
    ax1.bar(df_summary["sampler"], df_summary["best_value"], color="#2ecc71", alpha=0.7)
    ax1.set_ylabel("Best Objective Value", fontsize=11)
    ax1.set_title("Best Performance", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Convergence ratios
    ax2.bar(
        df_summary["sampler"],
        df_summary["convergence_ratio"],
        color="#e74c3c",
        alpha=0.7,
    )
    ax2.set_ylabel("Convergence Ratio (lower = faster)", fontsize=11)
    ax2.set_title("Convergence Speed", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    summary_plot = output_dir / "summary_comparison.png"
    plt.savefig(summary_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary plot: {summary_plot}")


def main():
    parser = argparse.ArgumentParser(
        description="Scientific sampler comparison for thesis"
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["qmc", "tpe", "cmaes"],
        help="Samplers to compare",
    )
    parser.add_argument(
        "--n-trials", type=int, default=500, help="Number of trials per sampler"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=673, help="Candidate pool size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--hamburg", action="store_true", help="Use Hamburg preselection"
    )
    parser.add_argument(
        "--KDR146", action="store_true", help="Use KDR_146 preselection"
    )

    args = parser.parse_args()

    preselection_flag = None
    if args.hamburg:
        preselection_flag = "--hamburg"
    elif args.KDR146:
        preselection_flag = "--KDR146"

    return run_sampler_comparison(
        samplers=args.samplers,
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        seed=args.seed,
        preselection_flag=preselection_flag,
        exp_description=f"Sampler comparison (n_trials={args.n_trials})",
    )


if __name__ == "__main__":
    sys.exit(main())
