#!/usr/bin/env python3
"""
Orchestrator for thesis-grade sampler evaluation.

1) Run multi-seed sampler comparisons on 'hamburg' and 'kdr100'
2) Compute best sampler per dataset and overall
3) Launch full adaptive runs (n_trials=2000) with best sampler on Hamburg and KDR100

Usage:
    python scripts/run_thesis_sampler_suite.py --seeds 42 43 44 45 46 --n-trials 500 --n-trials-full 2000 --n-candidates 673
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
    print(f"RUN: {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def choose_best_sampler(results_dir: Path):
    # results_dir contains per-dataset subfolders with summary.csv in each; read them all
    summaries = []
    for dataset_dir in (results_dir).glob("*/"):
        summary_csv = dataset_dir / "summary.csv"
        if summary_csv.exists():
            summaries.append(summary_csv)

    if not summaries:
        raise RuntimeError(f"No summary files found in {results_dir} subfolders")

    df_all = []
    for s in summaries:
        try:
            df = pd.read_csv(s)
            # Add dataset label from parent dir
            dataset = s.parent.name
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--n-trials", type=int, default=500, help="Trials per sampler in comparison"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hamburg", "kdr100"],
        help="Datasets to compare on",
    )
    parser.add_argument("--samplers", nargs="+", default=["qmc", "tpe", "cmaes"])
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument(
        "--n-trials-full", type=int, default=2000, help="Trials for full adaptive runs"
    )
    parser.add_argument("--n-candidates", type=int, default=673)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    suite_dir = OUT_BASE / f"sampler_thesis_suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # 1) Run compare_samplers_multi_seed.py
    out_arg = f"--output {suite_dir}"
    seeds_arg = " ".join(str(s) for s in args.seeds)
    samplers_arg = " ".join(args.samplers)
    datasets_arg = " ".join(args.datasets)

    compare_cmd = f"python scripts/compare_samplers_multi_seed.py --samplers {samplers_arg} --seeds {seeds_arg} --n-trials {args.n_trials} --datasets {datasets_arg} --sequential --output {suite_dir}"
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

    # Hamburg full run
    run_name_h = f"suite_full_{best}_hamburg_{timestamp}"
    cmd_h = f"PYTHONPATH=. {wrapper} --env dataselector -- python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {args.n_candidates} --sampler {best} --seed {args.seeds[0]} --hamburg"
    print(f"Launching full Hamburg run: {cmd_h}")
    run_cmd(cmd_h)

    # KDR100 full run (no preselection)
    run_name_k = f"suite_full_{best}_kdr100_{timestamp}"
    cmd_k = f"PYTHONPATH=. {wrapper} --env dataselector -- python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {args.n_candidates} --sampler {best} --seed {args.seeds[0]}"
    print(f"Launching full KDR100 run: {cmd_k}")
    run_cmd(cmd_k)

    print("\n=== SUITE COMPLETE ===")
    print(f"Results and artifacts: {suite_dir}")
