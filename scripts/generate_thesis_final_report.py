#!/usr/bin/env python3
"""Generate final thesis report from both full adaptive runs."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "runs"

# Run directories
hamburg_run = OUT_DIR / "20260117_T160726_adaptive_full"
kdr100_run = OUT_DIR / "20260117_T160740_adaptive_full"


def load_and_analyze(run_dir, name):
    """Load trials and bootstrap results from a run."""
    trials_csv = run_dir / "results" / "trials.csv"
    bootstrap_csv = run_dir / "results" / "bootstrap_results_summary.csv"
    best_trial_json = run_dir / "results" / "best_trial.json"

    df_trials = pd.read_csv(trials_csv)
    df_trials = df_trials[df_trials["state"] == "TrialState.COMPLETE"]

    best_value = df_trials["value"].max()
    best_trial_num = df_trials[df_trials["value"] == best_value]["trial_number"].iloc[0]
    mean_value = df_trials["value"].mean()
    std_value = df_trials["value"].std()
    median_value = df_trials["value"].median()

    # 99% convergence trial
    cummax = df_trials["value"].expanding().max()
    threshold_idx = (
        (cummax >= (best_value * 0.99)).idxmax()
        if (cummax >= (best_value * 0.99)).any()
        else len(df_trials) - 1
    )
    convergence_trial = int(df_trials.iloc[threshold_idx]["trial_number"])
    convergence_ratio = convergence_trial / len(df_trials)

    # Load best trial params
    with open(best_trial_json) as f:
        best_params = json.load(f)

    # Bootstrap summary
    df_boot = pd.read_csv(bootstrap_csv, index_col=0)
    boot_mean = (
        float(df_boot.loc["mean", "best_value"])
        if "best_value" in df_boot.columns
        else mean_value
    )
    boot_std = (
        float(df_boot.loc["std", "best_value"])
        if "best_value" in df_boot.columns
        else std_value
    )
    boot_ci_lo = (
        float(df_boot.loc["ci_lo", "best_value"])
        if "best_value" in df_boot.columns
        else np.nan
    )
    boot_ci_hi = (
        float(df_boot.loc["ci_hi", "best_value"])
        if "best_value" in df_boot.columns
        else np.nan
    )

    return {
        "name": name,
        "best_value": best_value,
        "best_trial": best_trial_num,
        "mean_value": mean_value,
        "median_value": median_value,
        "std_value": std_value,
        "convergence_trial": convergence_trial,
        "convergence_ratio": convergence_ratio,
        "n_trials": len(df_trials),
        "best_params": best_params,
        "boot_mean": boot_mean,
        "boot_std": boot_std,
        "boot_ci_lo": boot_ci_lo,
        "boot_ci_hi": boot_ci_hi,
    }


def main(hamburg_run: str | Path = None, kdr100_run: str | Path = None) -> int:
    """Generate final thesis report by analyzing two run directories.

    This function can be called programmatically or via CLI. Defaults use
    historical run paths but callers may override them.
    """
    hamb = Path(hamburg_run) if hamburg_run is not None else (ROOT / "outputs" / "runs" / "20260117_T160726_adaptive_full")
    kdr = Path(kdr100_run) if kdr100_run is not None else (ROOT / "outputs" / "runs" / "20260117_T160740_adaptive_full")

    hamburg = load_and_analyze(hamb, "Hamburg (800 candidates)")
    kdr100 = load_and_analyze(kdr, "KDR100 (673 candidates)")

    # Generate markdown report
    report = []
    report.append("# Thesis Final Report: CMA-ES Optimization on Hamburg & KDR100")
    report.append("")
    report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")
    report.append("")

    report.append("## Executive Summary")
    report.append("")
    report.append(
        f"- **Best overall value**: {max(hamburg['best_value'], kdr100['best_value']):.6f}"
    )
    report.append(
        f"  - Hamburg: {hamburg['best_value']:.6f} @ Trial #{int(hamburg['best_trial'])}"
    )
    report.append(
        f"  - KDR100: {kdr100['best_value']:.6f} @ Trial #{int(kdr100['best_trial'])}"
    )
    report.append("- **Sampler**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)")
    report.append("- **Trials per run**: 2000 (Optuna, with 200 bootstrap resamples)")
    report.append("- **Exploration**: Sobol (20 samples) → Fine Sweep (5 bounds) → Optuna")
    report.append("")

    # Detailed sections
    report.append("## Detailed Results")
    report.append("")

    for data in [hamburg, kdr100]:
        report.append(f"### {data['name']}")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Best Value | {data['best_value']:.6f} |")
        report.append(f"| Best Trial | #{int(data['best_trial'])} |")
        report.append(
            f"| Mean ± Std | {data['mean_value']:.6f} ± {data['std_value']:.6f} |"
        )
        report.append(f"| Median | {data['median_value']:.6f} |")
        report.append(
            f"| 99% Convergence Trial | #{int(data['convergence_trial'])} ({data['convergence_ratio']:.1%}) |"
        )
        report.append(
            f"| Bootstrap Mean ± 95% CI | {data['boot_mean']:.6f} ± [{data['boot_ci_lo']:.6f}, {data['boot_ci_hi']:.6f}] |"
        )
        report.append("")

        report.append("**Best Trial Parameters**:")
        report.append(f"  - Weight a (tile density): {data['best_params'].get('a', 'N/A')}")
        report.append(
            f"  - Weight b (spatial spread): {data['best_params'].get('b', 'N/A')}"
        )
        report.append(
            f"  - Weight c (temporal balance): {data['best_params'].get('c', 'N/A')}"
        )
        report.append(
            f"  - Min distance: {data['best_params'].get('min_distance_km', 'N/A')} km"
        )
        report.append(f"  - N samples: {data['best_params'].get('n_samples', 'N/A')}")
        report.append("")

    # Comparative analysis
    report.append("## Comparative Analysis")
    report.append("")
    report.append("| Dataset | Hamburg | KDR100 | Difference |")
    report.append("|---------|---------|--------|-----------|")
    report.append(
        f"| Best Value | {hamburg['best_value']:.6f} | {kdr100['best_value']:.6f} | {abs(hamburg['best_value'] - kdr100['best_value']):.6f} |"
    )

    # Save report
    report_file = ROOT / "outputs" / "THESIS_FINAL_REPORT.md"
    report_file.write_text("\n".join(report))
    print(f"✅ Report written: {report_file}")

    # Also save as CSV summary
    summary_data = {
        "dataset": ["Hamburg", "KDR100"],
        "best_value": [hamburg["best_value"], kdr100["best_value"]],
        "mean_value": [hamburg["mean_value"], kdr100["mean_value"]],
        "std_value": [hamburg["std_value"], kdr100["std_value"]],
        "convergence_trial": [hamburg["convergence_trial"], kdr100["convergence_trial"]],
        "convergence_ratio": [hamburg["convergence_ratio"], kdr100["convergence_ratio"]],
        "bootstrap_ci_lo": [hamburg["boot_ci_lo"], kdr100["boot_ci_lo"]],
        "bootstrap_ci_hi": [hamburg["boot_ci_hi"], kdr100["boot_ci_hi"]],
    }
    df_summary = pd.DataFrame(summary_data)
    summary_csv = ROOT / "outputs" / "THESIS_FINAL_SUMMARY.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"✅ Summary CSV written: {summary_csv}")

    print("\n" + "=" * 70)
    print("THESIS FINAL RESULTS")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())