#!/usr/bin/env python3
"""
Compare samplers across multiple random seeds and optionally multiple datasets.

Produces:
 - CSV summary with per-run best values
 - Statistical tests (pairwise Mann-Whitney U)
 - Effect sizes (Cohen's d)
 - Plots (boxplots, convergence summary)

Usage:
  ./scripts/exec_in_env.sh --env dataselector -- python scripts/compare_samplers_multi_seed.py --samplers qmc tpe cmaes --seeds 42 43 44 45 46 --n-trials 500 --hamburg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Defer matplotlib backend selection and pyplot import to runtime to avoid
# module-level side-effects that break importability and lint rules.

# For statistical tests
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

ROOT = Path(__file__).resolve().parents[1]
# Do not modify sys.path at import time; use runtime imports or PYTHONPATH instead
CSV_META_PATH = ROOT / "data" / "new_all_tiles.csv"


def run_single_optuna(
    sampler: str,
    seed: int,
    n_trials: int,
    n_candidates: int,
    preselection_flag: str,
    exp_desc: str,
    dataset: str = None,
    fixed_n_samples: int | None = None,
    constrain_a: tuple | None = None,
    constrain_b: tuple | None = None,
    constrain_c: tuple | None = None,
    constrain_min_dist: tuple | None = None,
) -> Dict:
    """Run `scripts/optuna_optimize.py` for one sampler/seed and return run metadata."""
    import time

    dataset_prefix = f"{dataset}_" if dataset else ""
    exp_name = f"{dataset_prefix}{sampler}_{n_trials}trials_s{seed}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "optuna_optimize.py"),
        "--n-trials",
        str(n_trials),
        "--n-candidates",
        str(n_candidates),
    ]

    # If a fixed n_samples was provided by autoscale, pass it directly; otherwise keep default range
    if fixed_n_samples is not None:
        cmd += ["--n-samples", str(fixed_n_samples)]
    else:
        cmd += ["--n-samples-min", "30", "--n-samples-max", "50"]

    # Add constrained bounds if provided (from autoscale optimization)
    if constrain_a is not None:
        cmd += ["--constrain-a-min", str(constrain_a[0]), "--constrain-a-max", str(constrain_a[1])]
    if constrain_b is not None:
        cmd += ["--constrain-b-min", str(constrain_b[0]), "--constrain-b-max", str(constrain_b[1])]
    if constrain_c is not None:
        cmd += ["--constrain-c-min", str(constrain_c[0]), "--constrain-c-max", str(constrain_c[1])]
    if constrain_min_dist is not None:
        cmd += ["--constrain-min-dist-min", str(constrain_min_dist[0]), "--constrain-min-dist-max", str(constrain_min_dist[1])]

    cmd += [
        "--sampler",
        sampler,
        "--seed",
        str(seed),
        "--exp-name",
        exp_name,
        "--exp-desc",
        f"{exp_desc} (sampler={sampler}, seed={seed})",
    ]
    if preselection_flag:
        cmd.append(preselection_flag)

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Run failed for {sampler} seed {seed}")

    # Find latest run dir
    run_dirs = sorted((ROOT / "outputs" / "runs").glob(f"*{exp_name}"))
    if not run_dirs:
        out = proc.stdout or ""
        err = proc.stderr or ""
        msg = f"No run dir found for {exp_name}\nSubprocess stdout:\n{out}\nSubprocess stderr:\n{err}"
        # common hint: missing optuna in environment
        if "ModuleNotFoundError" in out or "ModuleNotFoundError" in err:
            msg += "\nHint: a ModuleNotFoundError was observed in the subprocess output; ensure required packages (e.g., optuna) are installed in the environment."
        raise FileNotFoundError(msg)
    run_dir = run_dirs[-1]
    trials_csv = run_dir / "results" / "trials.csv"

    # Retry loop for output integrity (filesystem latency)
    for _ in range(5):
        if trials_csv.exists() and trials_csv.stat().st_size > 0:
            break
        time.sleep(1)

    if not trials_csv.exists():
        raise FileNotFoundError(f"trials.csv missing in {run_dir}")

    df = pd.read_csv(trials_csv)
    df = df[df["value"].notna()]

    # Normalize trial number column names across different optuna versions/formats
    for col in ("trial_number", "number", "trial"):
        if col in df.columns:
            df = df.rename(columns={col: "trial_number"})
            break

    best_val = float(df["value"].max()) if len(df) > 0 else float("nan")
    if len(df) > 0 and "trial_number" in df.columns:
        best_trial = int(df.loc[df["value"].idxmax(), "trial_number"])
    else:
        best_trial = -1
    cumulative_best = (
        df["value"].expanding().max() if len(df) > 0 else pd.Series(dtype=float)
    )
    threshold_idx = (
        (cumulative_best >= (best_val * 0.99)).idxmax()
        if (len(df) > 0 and (cumulative_best >= (best_val * 0.99)).any())
        else (len(df) - 1)
    )

    return {
        "sampler": sampler,
        "seed": seed,
        "n_trials": len(df),
        "best_value": best_val,
        "best_trial": best_trial,
        "mean_value": float(df["value"].mean()) if len(df) > 0 else float("nan"),
        "std_value": float(df["value"].std()) if len(df) > 0 else float("nan"),
        "convergence_trial": int(threshold_idx),
        "convergence_ratio": (
            float(threshold_idx / len(df)) if len(df) > 0 else float("nan")
        ),
        "run_dir": str(run_dir),
    }


def compare_and_analyze(results_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw
    raw_csv = out_dir / "per_run_results.csv"
    results_df.to_csv(raw_csv, index=False)
    print(f"Saved per-run results: {raw_csv}")

    # Group by sampler
    grouped = results_df.groupby("sampler")["best_value"].apply(list).to_dict()

    # Basic summary
    summary = (
        results_df.groupby("sampler")["best_value"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    # Bootstrap 95% CI for mean & median per sampler
    rng = np.random.default_rng(42)
    ci_rows = []
    for sampler, vals in grouped.items():
        arr = np.array(vals)
        boot_means = []
        boot_medians = []
        for _ in range(2000):
            resample = rng.choice(arr, size=arr.size, replace=True)
            boot_means.append(resample.mean())
            boot_medians.append(np.median(resample))
        mean_lo, mean_hi = np.percentile(boot_means, [2.5, 97.5])
        med_lo, med_hi = np.percentile(boot_medians, [2.5, 97.5])
        ci_rows.append(
            {
                "sampler": sampler,
                "mean_ci_lo": mean_lo,
                "mean_ci_hi": mean_hi,
                "median_ci_lo": med_lo,
                "median_ci_hi": med_hi,
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on="sampler")
    summary_file = out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

    # Statistical tests: pairwise Mann-Whitney U
    stats = []
    samplers = list(grouped.keys())
    for i in range(len(samplers)):
        for j in range(i + 1, len(samplers)):
            s1 = samplers[i]
            s2 = samplers[j]
            a = np.array(grouped[s1])
            b = np.array(grouped[s2])
            if mannwhitneyu is None:
                pval = float("nan")
            else:
                try:
                    u = mannwhitneyu(a, b, alternative="two-sided")
                    pval = float(u.pvalue)
                except Exception:
                    pval = float("nan")
            # Cohen's d
            pooled_std = (
                np.sqrt(
                    (
                        (a.size - 1) * a.std(ddof=1) ** 2
                        + (b.size - 1) * b.std(ddof=1) ** 2
                    )
                    / (a.size + b.size - 2)
                )
                if (a.size > 1 and b.size > 1)
                else float("nan")
            )
            cohen_d = (
                (a.mean() - b.mean()) / pooled_std
                if pooled_std and not np.isnan(pooled_std) and pooled_std != 0
                else float("nan")
            )
            stats.append(
                {"sampler1": s1, "sampler2": s2, "pvalue": pval, "cohens_d": cohen_d}
            )

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "pairwise_stats.csv", index=False)
    print(f"Saved pairwise statistics: {out_dir / 'pairwise_stats.csv'}")

    # Plots: boxplot of best values
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [grouped[s] for s in samplers]
    ax.boxplot(data, tick_labels=[s.upper() for s in samplers], patch_artist=True)
    ax.set_title("Best Value Distribution per Sampler (across seeds)")
    ax.set_ylabel("Objective Value")

    # Annotate pairwise p-values
    y_max = max([np.max(d) for d in data])
    y_min = min([np.min(d) for d in data])
    y = y_max + (y_max - y_min) * 0.05
    for idx, row in stats_df.iterrows() if "stats_df" in locals() else []:
        s1 = row["sampler1"]
        s2 = row["sampler2"]
        p = row["pvalue"]
        if not np.isnan(p):
            txt = f"p={p:.3f}"
            ax.text(
                0.5,
                y + idx * (y_max - y_min) * 0.02,
                f"{row['sampler1'].upper()} vs {row['sampler2'].upper()}: {txt}",
                fontsize=8,
            )

    plt.tight_layout()
    bp = out_dir / "best_value_boxplot.png"
    plt.savefig(bp, dpi=300)
    plt.close()
    print(f"Saved boxplot: {bp}")

    # Convergence summary: compile median cumulative best across seeds for each sampler
    # Load per-run histories and compute median curve
    median_curves = {}
    for sampler in samplers:
        # Collect cumulative arrays
        cumuls = []
        for idx, row in results_df[results_df["sampler"] == sampler].iterrows():
            trials_csv = Path(row["run_dir"]) / "results" / "trials.csv"
            if trials_csv.exists():
                df = pd.read_csv(trials_csv)
                df = df[df["value"].notna()]
                cumuls.append(df["value"].expanding().max().values)
        if cumuls:
            # pad to same length
            maxlen = max(len(a) for a in cumuls)
            arr = np.array(
                [
                    np.pad(a, (0, maxlen - len(a)), constant_values=np.nan)
                    for a in cumuls
                ]
            )
            median_curves[sampler] = np.nanmedian(arr, axis=0)

    if median_curves:
        fig, ax = plt.subplots(figsize=(10, 6))
        for s, curve in median_curves.items():
            ax.plot(curve, label=s.upper())
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Median Best Objective")
        ax.set_title("Median Convergence Curves (across seeds)")
        ax.legend()
        plt.tight_layout()
        conv_file = out_dir / "median_convergence.png"
        plt.savefig(conv_file, dpi=300)
        plt.close()
        print(f"Saved convergence plot: {conv_file}")

    return {
        "summary_file": str(summary_file),
        "pairwise_stats": str(out_dir / "pairwise_stats.csv"),
        "plots": [str(bp), str(conv_file) if median_curves else ""],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare samplers across multiple seeds and datasets"
    )
    parser.add_argument("--samplers", nargs="+", default=["qmc", "tpe", "cmaes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=None)  # Read dynamically from CSV if not set
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hamburg", "kdr100", "full"],
        default=None,
        help="Datasets to run on",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run sequentially (default)"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Fixed n_samples to use for all runs (overrides n-samples-min/n-samples-max range)",
    )
    parser.add_argument(
        "--constrain-a-min",
        type=float,
        default=None,
        help="Constrain alpha (a) lower bound (from autoscale)",
    )
    parser.add_argument(
        "--constrain-a-max",
        type=float,
        default=None,
        help="Constrain alpha (a) upper bound",
    )
    parser.add_argument(
        "--constrain-b-min",
        type=float,
        default=None,
        help="Constrain beta (b) lower bound",
    )
    parser.add_argument(
        "--constrain-b-max",
        type=float,
        default=None,
        help="Constrain beta (b) upper bound",
    )
    parser.add_argument(
        "--constrain-c-min",
        type=float,
        default=None,
        help="Constrain gamma (c) lower bound",
    )
    parser.add_argument(
        "--constrain-c-max",
        type=float,
        default=None,
        help="Constrain gamma (c) upper bound",
    )
    parser.add_argument(
        "--constrain-min-dist-min",
        type=int,
        default=None,
        help="Constrain min_distance lower bound",
    )
    parser.add_argument(
        "--constrain-min-dist-max",
        type=int,
        default=None,
        help="Constrain min_distance upper bound",
    )
    args = parser.parse_args()

    # Configure matplotlib backend and make pyplot available to helper functions
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # expose plt at module level so functions can call it
    globals()["plt"] = plt

    # determine dataset/config
    datasets = args.datasets or ["full"]

    timestamp = datetime.now().strftime("%Y%m%d_T%H%M%S")
    # Use --output if provided, otherwise default to outputs/runs/
    if args.output:
        global_out_dir = Path(args.output)
    else:
        global_out_dir = ROOT / "outputs" / "runs" / f"sampler_multi_{timestamp}"
    global_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset in datasets:
        print(f"\n=== Running dataset: {dataset} ===\n")
        preselection_flag = "--hamburg" if dataset == "hamburg" else None
        if args.n_candidates is None:
            try:
                df_meta = pd.read_csv(CSV_META_PATH)
                n_candidates = len(df_meta)  # Dynamically from CSV
                print(f"[INFO] Auto-detected n_candidates from CSV: {n_candidates}")
            except FileNotFoundError:
                n_candidates = 676
                print(f"WARNING: {CSV_META_PATH} not found, using default {n_candidates}")
        else:
            n_candidates = args.n_candidates

        dataset_out = global_out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        for sampler in args.samplers:
            for seed in args.seeds:
                print(f"Starting run: dataset={dataset} sampler={sampler} seed={seed}")
                
                # Build constraint tuples if provided
                constrain_a = (args.constrain_a_min, args.constrain_a_max) if args.constrain_a_min is not None else None
                constrain_b = (args.constrain_b_min, args.constrain_b_max) if args.constrain_b_min is not None else None
                constrain_c = (args.constrain_c_min, args.constrain_c_max) if args.constrain_c_min is not None else None
                constrain_min_dist = (args.constrain_min_dist_min, args.constrain_min_dist_max) if args.constrain_min_dist_min is not None else None
                
                meta = run_single_optuna(
                    sampler,
                    seed,
                    args.n_trials,
                    n_candidates,
                    preselection_flag,
                    f"Multi-seed comparison ({dataset})",
                    dataset=dataset,
                    fixed_n_samples=args.n_samples,
                    constrain_a=constrain_a,
                    constrain_b=constrain_b,
                    constrain_c=constrain_c,
                    constrain_min_dist=constrain_min_dist,
                )
                meta["dataset"] = dataset
                all_results.append(meta)

    df_results = pd.DataFrame(all_results)

    analysis = compare_and_analyze(df_results, global_out_dir)
    print("Analysis complete:", analysis)

    # Determine best sampler (mean best_value across seeds)
    try:
        best_sampler = df_results.groupby("sampler")["best_value"].mean().idxmax()
        best_score = float(df_results.groupby("sampler")["best_value"].mean().max())
    except Exception:
        best_sampler = None
        best_score = None

    selected = {
        "best": best_sampler,
        "metric": "mean_best_value",
        "score": best_score,
        "n_trials": int(args.n_trials),
        "seeds": list(args.seeds),
        "datasets": datasets,
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(global_out_dir),
    }

    # Persist selected sampler artifact to a canonical location for the monitor
    try:
        sel_file = ROOT / "outputs" / "selected_sampler.json"
        sel_file.write_text(
            pd.json.dumps(selected)
            if hasattr(pd, "json")
            else __import__("json").dumps(selected, indent=2)
        )
        print(f"Wrote selected sampler artifact: {sel_file}")
    except Exception as e:
        print(f"Warning: could not write selected_sampler.json: {e}")

    # Also write inside the experiment-specific output folder for convenience
    try:
        (global_out_dir / "selected_sampler.json").write_text(
            __import__("json").dumps(selected, indent=2)
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
