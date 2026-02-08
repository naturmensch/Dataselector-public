#!/usr/bin/env python3
"""
Thesis Sampler Suite Workflow — Comprehensive Sampler Evaluation

Orchestrates thesis-grade sampler evaluation:
1. Optional: Autoscale for n_samples determination
2. Multi-seed sampler comparison across datasets
3. Best sampler selection based on performance
4. Full adaptive runs with best sampler on Hamburg and KDR100

Migration from: scripts/run_thesis_sampler_suite.py
Author: Phase 4 Migration
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dataselector.cli_decorators import cli_command
from dataselector.runtime import activate_repro_mode, write_run_metadata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import pandas as pd


THESIS_SAMPLER_SUITE_ARGS = {
    "seeds": {
        "type": int,
        "nargs": "+",
        "default": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        "help": "Random seeds for reproducibility (default: 10 seeds for thesis-grade validation)",
    },
    "n_trials": {
        "type": int,
        "default": 1000,
        "help": "Trials per sampler in comparison (default: 1000 per convergence analysis: 99%% optimum at ~650 trials; 1000 provides thesis-grade robustness)",
    },
    "datasets": {
        "type": str,
        "nargs": "+",
        "default": ["hamburg", "kdr100"],
        "help": "Datasets to compare on (default: hamburg + kdr100 for representative comparison)",
    },
    "samplers": {
        "type": str,
        "nargs": "+",
        "default": ["qmc", "tpe", "cmaes"],
        "help": "Samplers to compare (default: QMC, TPE, CMA-ES)",
    },
    "sequential": {
        "type": bool,
        "action": "store_true",
        "help": "Run sequentially",
    },
    "n_trials_full": {
        "type": int,
        "default": 2000,
        "help": "Trials for full adaptive runs",
    },
    "n_candidates": {
        "type": int,
        "default": None,
        "help": "Number of candidate tiles",
    },
    "autoscale": {
        "type": bool,
        "action": "store_true",
        "help": "Run optuna_autoscale.py to determine best n_samples before running sampler suite",
    },
    "execution_profile": {
        "type": str,
        "choices": ["default", "thesis_repro"],
        "default": "default",
        "help": "Runtime execution profile",
    },
}


def run_cmd(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    """
    Run a shell command and fail fast on non-zero exit.

    Parameters
    ----------
    cmd : list[str]
        Command argument vector to execute
    cwd : Path | None
        Working directory for command execution
    """
    print(f"RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, env=env)

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def map_sampler_for_adaptive_pipeline(best_sampler: str) -> tuple[str, str]:
    """Map suite sampler IDs to adaptive exploration and Optuna sampler args."""
    method = str(best_sampler).strip().lower()
    mapping: dict[str, tuple[str, str]] = {
        "qmc": ("sobol", "QMCSampler"),
        "tpe": ("lhs", "TPESampler"),
        "cmaes": ("lhs", "CmaEsSampler"),
    }
    if method not in mapping:
        raise RuntimeError(
            f"Unsupported best sampler '{best_sampler}'. "
            "Expected one of: qmc, tpe, cmaes."
        )
    return mapping[method]


def choose_best_sampler(results_dir: Path) -> tuple[str, "pd.DataFrame"]:
    """
    Choose the best sampler based on performance summaries.

    Parameters
    ----------
    results_dir : Path
        Directory containing sampler comparison results

    Returns
    -------
    tuple[str, pd.DataFrame]
        Best sampler name and summary table

    Raises
    ------
    RuntimeError
        If no summary files found or could be read
    """
    import pandas as pd

    # Try to read per-dataset summary files first
    summaries = []
    for dataset_dir in results_dir.glob("*/"):
        summary_csv = dataset_dir / "summary.csv"
        if summary_csv.exists():
            summaries.append(summary_csv)

    # Fallback: some analysis scripts write a global summary.csv at results_dir
    if not summaries:
        global_summary = results_dir / "summary.csv"
        if global_summary.exists():
            summaries = [global_summary]

    if not summaries:
        raise RuntimeError(
            f"No summary files found in {results_dir} subfolders or {results_dir}/summary.csv"
        )

    df_all = []
    for s in summaries:
        try:
            df = pd.read_csv(s)
            # If this is a global summary, dataset column may be missing
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


def run_thesis_sampler_suite(
    seeds: list[int] | None = None,
    n_trials: int = 1000,
    datasets: list[str] | None = None,
    samplers: list[str] | None = None,
    sequential: bool = True,
    n_trials_full: int = 2000,
    n_candidates: int | None = None,
    autoscale: bool = True,
    execution_profile: str = "default",
) -> Path:
    """
    Execute thesis-grade sampler evaluation suite.

    Parameters
    ----------
    seeds : list[int] | None
        Random seeds for reproducibility (default: 10 seeds)
    n_trials : int
        Trials per sampler in comparison (default: 1000)
    datasets : list[str] | None
        Datasets to compare on (default: ['hamburg', 'kdr100'])
    samplers : list[str] | None
        Samplers to compare (default: ['qmc', 'tpe', 'cmaes'])
    sequential : bool
        Run sequentially (default: True)
    n_trials_full : int
        Trials for full adaptive runs (default: 2000)
    n_candidates : int | None
        Candidate pool size (None = read from CSV)
    autoscale : bool
        Run autoscale before suite (default: True)

    Returns
    -------
    Path
        Suite output directory
    """
    import pandas as pd

    # Set defaults
    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    if datasets is None:
        datasets = ["hamburg", "kdr100"]
    if samplers is None:
        samplers = ["qmc", "tpe", "cmaes"]

    OUT_BASE = ROOT / "outputs" / "runs"
    primary_seed = seeds[0]
    runtime_state = activate_repro_mode(profile=execution_profile, seed=primary_seed)
    child_env = dict(os.environ)

    # Dynamically read n_candidates from CSV if not set
    if n_candidates is None:
        csv_path = ROOT / "data" / "new_all_tiles.csv"
        if csv_path.exists():
            n_candidates = len(pd.read_csv(csv_path))
            print(f"Dynamically determined n_candidates={n_candidates} from {csv_path}")
        else:
            print(f"WARNING: {csv_path} not found, using default 676")
            n_candidates = 676

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    suite_dir = OUT_BASE / f"sampler_thesis_suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # 1) Optionally run autoscale to determine n_samples and hyperparams
    best_n_samples = None
    constrain_bounds = {}
    if autoscale:
        print(
            "Running autoscale to determine best n_samples and hyperparams (this may take some time)..."
        )

        from dataselector.workflows.optuna_autoscale import (
            load_or_create_data,
            run_autoscale,
        )

        # Load data
        features, metadata = load_or_create_data(
            out_dir=Path("outputs"), n=n_candidates, dim=256, seed=primary_seed
        )

        try:
            # Run autoscale directly
            summary_file, best_json_path = run_autoscale(
                n_trials_per_stage=[20, 40, 80, 160],
                stages_samples=[50, 100, 300, len(features)],
                features=features,
                metadata=metadata,
                patience=2,
                out_dir=Path("outputs"),
            )
            print(f"Autoscale complete: {best_json_path}")

            # Read full best JSON to extract hyperparams
            best_json = Path("outputs") / "optuna_autoscale_best_latest.json"
            try:
                if best_json.exists():
                    data = json.loads(best_json.read_text())
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

                    print(
                        f"Constrained bounds: a=[{constrain_bounds['a_min']:.3f}, {constrain_bounds['a_max']:.3f}], "
                        f"b=[{constrain_bounds['b_min']:.3f}, {constrain_bounds['b_max']:.3f}], "
                        f"c=[{constrain_bounds['c_min']:.3f}, {constrain_bounds['c_max']:.3f}], "
                        f"min_dist=[{constrain_bounds['min_dist_min']}, {constrain_bounds['min_dist_max']}]"
                    )
                else:
                    # try to read simple n_samples text file
                    sel_file = (
                        Path("outputs") / "optuna_autoscale_selected_n_samples.txt"
                    )
                    if sel_file.exists():
                        best_n_samples = int(sel_file.read_text().strip())
                        print(
                            f"Autoscale selected n_samples (from text file)={best_n_samples}"
                        )
            except Exception as e:
                print(f"Warning: could not parse autoscale output: {e}")
        except Exception as e:
            print(f"Error running autoscale: {e}")
            print("Proceeding without autoscale results.")

    # 1b) Run compare-samplers workflow
    compare_cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "compare-samplers",
        "--samplers",
        *samplers,
        "--seeds",
        *[str(s) for s in seeds],
        "--n-trials",
        str(n_trials),
        "--datasets",
        *datasets,
        "--output",
        str(suite_dir),
        "--n-candidates",
        str(n_candidates),
    ]
    if sequential:
        compare_cmd.append("--sequential")
    if best_n_samples is not None:
        compare_cmd.extend(["--n-samples", str(best_n_samples)])
    if constrain_bounds:
        compare_cmd.extend(
            [
                "--constrain-a-min",
                str(constrain_bounds["a_min"]),
                "--constrain-a-max",
                str(constrain_bounds["a_max"]),
                "--constrain-b-min",
                str(constrain_bounds["b_min"]),
                "--constrain-b-max",
                str(constrain_bounds["b_max"]),
                "--constrain-c-min",
                str(constrain_bounds["c_min"]),
                "--constrain-c-max",
                str(constrain_bounds["c_max"]),
                "--constrain-min-dist-min",
                str(constrain_bounds["min_dist_min"]),
                "--constrain-min-dist-max",
                str(constrain_bounds["min_dist_max"]),
            ]
        )

    run_cmd(compare_cmd, env=child_env)

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
        raise RuntimeError("Best sampler selection failed") from e

    # 3) Launch full adaptive runs with best sampler: Hamburg and KDR100
    # Hamburg full run
    explore_sampler, optuna_sampler = map_sampler_for_adaptive_pipeline(best)
    cmd_h = [
        sys.executable,
        "-m",
        "dataselector",
        "adaptive-pipeline",
        "--n-trials",
        str(n_trials_full),
        "--n-candidates",
        str(n_candidates),
        "--sampler",
        explore_sampler,
        "--optuna-sampler",
        optuna_sampler,
        "--seed",
        str(seeds[0]),
        "--hamburg",
    ]
    print(f"Launching full Hamburg run: {' '.join(cmd_h)}")
    run_cmd(cmd_h, env=child_env)

    # KDR100 full run (no preselection)
    cmd_k = [
        sys.executable,
        "-m",
        "dataselector",
        "adaptive-pipeline",
        "--n-trials",
        str(n_trials_full),
        "--n-candidates",
        str(n_candidates),
        "--sampler",
        explore_sampler,
        "--optuna-sampler",
        optuna_sampler,
        "--seed",
        str(seeds[0]),
    ]
    print(f"Launching full KDR100 run: {' '.join(cmd_k)}")
    run_cmd(cmd_k, env=child_env)

    print("\n=== SUITE COMPLETE ===")
    print(f"Results and artifacts: {suite_dir}")
    write_run_metadata(
        output_dir=suite_dir,
        execution_profile=execution_profile,
        seed=primary_seed,
        runtime_state=runtime_state,
        extra={
            "n_trials": n_trials,
            "n_trials_full": n_trials_full,
            "n_candidates": n_candidates,
            "autoscale": autoscale,
            "datasets": datasets,
            "samplers": samplers,
        },
    )

    return suite_dir


@cli_command(
    "thesis-sampler-suite",
    help="Thesis-grade sampler evaluation suite with multi-seed comparison",
    args=THESIS_SAMPLER_SUITE_ARGS,
)
def main(
    seeds: list[int] | None = None,
    n_trials: int = 1000,
    datasets: list[str] | None = None,
    samplers: list[str] | None = None,
    sequential: bool = False,
    n_trials_full: int = 2000,
    n_candidates: int | None = None,
    autoscale: bool = True,
    execution_profile: str = "default",
) -> int:
    """CLI entry point for thesis sampler suite."""

    # Apply defaults if None
    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    if datasets is None:
        datasets = ["hamburg", "kdr100"]
    if samplers is None:
        samplers = ["qmc", "tpe", "cmaes"]

    suite_dir = run_thesis_sampler_suite(
        seeds=seeds,
        n_trials=n_trials,
        datasets=datasets,
        samplers=samplers,
        sequential=sequential,
        n_trials_full=n_trials_full,
        n_candidates=n_candidates,
        autoscale=autoscale,
        execution_profile=execution_profile,
    )

    print(f"\n✅ Thesis sampler suite completed. Results in: {suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
