#!/usr/bin/env python3
"""
Adaptive Pipeline Workflow — Multi-Stage Optimization Pipeline

Orchestrates the complete adaptive parameter search pipeline:
1. Exploration (LHS/Sobol sampling)
2. Fine Sweep (adaptive bounds)
3. Optimization (Optuna)
4. Validation (Bootstrap)

Migration from: scripts/run_adaptive_pipeline.py
Author: Phase 3R Migration
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dataselector.cli_decorators import cli_command

# Project imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


def run_cmd(cmd: str) -> None:
    """Execute a shell command, printing output; do not fail on nonzero exit."""
    print(f"🔧 Running: {cmd}")
    subprocess.call(cmd, shell=True)


def run_cmd_safe(cmd: str) -> None:
    """Execute a shell command and fail fast on nonzero exit."""
    print(f"🔧 Running (strict): {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise SystemExit(f"Command failed with exit code {ret}: {cmd}")


def run_adaptive_pipeline(
    experiment_name: str = "adaptive_pipeline",
    csv_path: str | Path | None = None,
    n_lhs: int | None = None,
    n_trials: int = 100,
    n_boot: int = 500,
    n_candidates: int | None = None,
    n_dimensions: int = 9,
    sampler: str = "lhs",
    optuna_sampler: str = "TPESampler",
    seed: int = 42,
    n_initial_strategy: str = "conservative",
    n_samples: int | None = None,
    n_samples_min: int | None = None,
    n_samples_max: int | None = None,
    fine_max_runs: int | None = None,
    skip_exploration: bool = False,
    skip_fine: bool = False,
    skip_optuna: bool = False,
    skip_bootstrap_injection: bool = False,
    continue_on_analysis_failure: bool = False,
    dry_run: bool = False,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
    hamburg: bool = False,
    KDR146: bool = False,
) -> Path:
    """
    Execute the full adaptive pipeline.

    Parameters
    ----------
    experiment_name : str
        Name for the experiment run
    csv_path : str | Path | None
        Path to tile metadata CSV
    n_lhs : int | None
        Number of LHS samples (None = adaptive)
    n_trials : int
        Number of Optuna trials
    n_boot : int
        Number of bootstrap resamples
    n_candidates : int | None
        Total candidate pool size (None = dataset size)
    n_dimensions : int
        Number of parameter dimensions
    sampler : str
        Exploration sampler: 'lhs' or 'sobol'
    optuna_sampler : str
        Optuna sampler class name
    seed : int
        Random seed
    n_initial_strategy : str
        Strategy for adaptive n_lhs: 'conservative', 'moderate', 'aggressive'
    n_samples : int | None
        Fixed n_samples for Optuna
    n_samples_min : int | None
        Min n_samples for Optuna range mode
    n_samples_max : int | None
        Max n_samples for Optuna range mode
    fine_max_runs : int | None
        Max runs for fine sweep
    skip_exploration : bool
        Skip exploration phase
    skip_fine : bool
        Skip fine sweep phase
    skip_optuna : bool
        Skip optimization phase
    skip_bootstrap_injection : bool
        Skip bootstrap-best config injection
    continue_on_analysis_failure : bool
        Continue pipeline on analysis errors
    dry_run : bool
        Dry run mode (no actual execution)
    pre_names : list[str] | None
        Pre-selected tile names
    pre_indices : list[int] | None
        Pre-selected tile indices
    hamburg : bool
        Convenience flag to add "Hamburg" to pre_names
    KDR146 : bool
        Convenience flag to add "KDR_146" to pre_names

    Returns
    -------
    Path
        Path to the experiment run directory
    """
    OUT = ROOT / "outputs" / "experiments"
    OUT.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid loading heavy dependencies during test imports
    from dataselector.pipeline.experiment_manager import ExperimentManager

    # Read metadata to get n_tiles
    n_tiles = None
    if csv_path is None:
        csv_path = ROOT / "data" / "new_all_tiles.csv"
    csv_path = Path(csv_path)

    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            n_tiles = len(df)
            print(f"📊 Loaded metadata: {n_tiles} tiles from {csv_path}")
        except Exception as e:
            print(f"⚠️  Could not read metadata from {csv_path}: {e}")
    else:
        print(f"⚠️  Metadata not found: {csv_path}")

    # Initialize experiment manager
    em = ExperimentManager(
        experiment_name,
        base_dir=OUT,
        metadata={
            "sampler": sampler,
            "optuna_sampler": optuna_sampler,
            "n_dimensions": n_dimensions,
            "seed": seed,
        },
    )
    em.save_manifest()

    # Expose run dir to sub-scripts via ENV
    os.environ["EXPERIMENT_RUN_DIR"] = str(em.run_dir)
    print(em.summary())

    # Import utilities
    from dataselector.pipeline.pipeline_utils import (
        compute_adaptive_n_initial,
        compute_fine_search_bounds,
        compute_optuna_bounds,
    )

    # INTEGRATION: Check for ExplorationPlan from benchmark-sampling
    # If benchmark was run, use its recommended sampler and n_initial settings
    exploration_plan = None
    plan_env = os.environ.get("DATASELECTOR_EXPLORATION_PLAN")
    if plan_env:
        try:
            import json

            plan_path = Path(plan_env)
            if plan_path.exists():
                with open(plan_path) as f:
                    plan_data = json.load(f)
                exploration_plan = plan_data
                print(
                    f"\n🎯 BENCHMARK INTEGRATION: Loaded exploration plan from {plan_env}"
                )
                print(f"   Recommended sampler: {plan_data.get('method', 'unknown')}")
                print(
                    f"   Recommended n_initial: {plan_data.get('n_initial_final', 'unknown')}"
                )

                # Override sampler with benchmark result (if not explicitly provided)
                if sampler is None:
                    sampler = plan_data.get("method")
                    print(f"   ✅ Using benchmark sampler: {sampler}")

                # Override n_lhs with benchmark result (if not explicitly provided)
                if n_lhs is None:
                    n_lhs = plan_data.get("n_initial_final")
                    print(f"   ✅ Using benchmark n_initial: {n_lhs}")
        except Exception as e:
            print(f"⚠️  Could not load exploration plan: {e}")
            exploration_plan = None

    # Enforce: sampler must be from benchmark OR CLI - NO FALLBACK
    if sampler is None:
        raise ValueError(
            "Sampler not specified. Either:\n"
            "1. Run 'dataselector benchmark-sampling' first and set DATASELECTOR_EXPLORATION_PLAN\n"
            "2. Provide --sampler {lhs,sobol} via CLI\n"
            "No hardcoded fallback (long-term solution)."
        )

    # Resolve n_candidates dynamically if not set
    if n_candidates is None:
        if n_tiles is not None:
            n_candidates = n_tiles
            print(f"📊 Using full dataset size for n_candidates: {n_candidates}")
        else:
            n_candidates = 673  # Fallback
            print(f"⚠️  Metadata not found; defaulting n_candidates to {n_candidates}")

    # Resolve n_lhs adaptively if not set
    if n_lhs is None:
        n_lhs = compute_adaptive_n_initial(
            n_dimensions, n_tiles=n_tiles, strategy=n_initial_strategy
        )
        print(f"📊 Adaptive n_lhs: {n_lhs} (strategy={n_initial_strategy})")
    else:
        print(f"📊 Using user-specified n_lhs: {n_lhs}")

    # LONG-TERM SOLUTION: No fallback - sampler must be provided explicitly
    if sampler is None:
        raise ValueError(
            "Sampler not specified. Either:\n"
            "1. Run 'dataselector benchmark-sampling' first (sets DATASELECTOR_EXPLORATION_PLAN)\n"
            "2. Provide --sampler {lhs,sobol} explicitly via CLI\n"
            "No hardcoded fallback is provided (long-term solution)."
        )

    # Adjust for Sobol sampler (prefer power of two)
    if sampler == "sobol":
        adjusted = _next_power_of_two(n_lhs)
        if adjusted != n_lhs:
            print(
                f"⚠ Using Sobol sampler: rounding n_lhs {n_lhs} -> next power of two {adjusted}"
            )
            n_lhs = adjusted

    # Persist final config
    try:
        em.save_config(
            "run",
            {
                "n_lhs": n_lhs,
                "n_trials": n_trials,
                "n_candidates": n_candidates,
                "sampler": sampler,
                "seed": seed,
            },
        )
        em.log(
            f"📊 Final config: n_lhs={n_lhs}, n_candidates={n_candidates} (sampler={sampler})"
        )
    except Exception as e:
        print(f"⚠ Could not persist final config to ExperimentManager: {e}")

    # Build pre-selection arguments
    pre_names_list = list(pre_names) if pre_names is not None else []
    if hamburg:
        pre_names_list.append("Hamburg")
    if KDR146:
        pre_names_list.append("KDR_146")
    pre_indices_list = list(pre_indices) if pre_indices is not None else []

    print(
        f"Using pre-selected names: {pre_names_list if pre_names_list else None}, "
        f"pre-selected indices: {pre_indices_list if pre_indices_list else None}"
    )

    # ========================================================================
    # COMPUTE INITIAL SPATIAL CONSTRAINT (min_distance_km)
    # ========================================================================
    from dataselector.pipeline.pipeline_utils import compute_min_distance_km

    print("=== Computing Initial Spatial Constraint ===")
    try:
        initial_min_distance = compute_min_distance_km(
            str(csv_path or DATA_CSV),
            strategy="median_nn",
            safety_margin=1.0,
        )
    except Exception as e:
        print(
            f"⚠️  Could not compute min_distance from data ({e}). "
            f"Using conservative default: 28.0 km"
        )
        initial_min_distance = 28.0

    # ========================================================================
    # PHASE 1: EXPLORATION (LHS/Sobol)
    # ========================================================================
    if skip_exploration:
        print("=== Phase 1: Exploration SKIPPED (--skip-exploration) ===")
    else:
        print(f"=== Phase 1: Exploration ({sampler.upper()}) ===")
        print(
            f"Running {sampler} with {n_lhs} samples (replacing old manual Coarse Grid)..."
        )

        from dataselector.workflows.tune_weights import run_exploration

        run_exploration(
            n_samples=n_lhs,
            sampler=sampler,
            seed=seed,
            metadata_path=str(csv_path),
            output_dir=OUT / "tuning_weights",
        )

    # ========================================================================
    # PHASE 2: FINE SWEEP (Adaptive Bounds)
    # ========================================================================
    pareto_lhs = OUT / "tuning_weights" / "pareto" / "pareto_solutions.csv"

    if skip_fine:
        print("=== Phase 2: Fine Sweep SKIPPED (--skip-fine) ===")
        # Prefer existing fine pareto, else fall back to exploration pareto
        if (OUT / "fine_sweep" / "pareto_solutions.csv").exists():
            pareto_fine = OUT / "fine_sweep" / "pareto_solutions.csv"
            print(f"Using existing fine pareto: {pareto_fine}")
        elif pareto_lhs.exists():
            pareto_fine = pareto_lhs
            print(f"No fine pareto found; using exploration pareto: {pareto_fine}")
        else:
            raise SystemExit(
                "No pareto available to proceed after skipping fine sweep; aborting"
            )
        fine_bounds = compute_fine_search_bounds(str(pareto_fine))
    else:
        if not pareto_lhs.exists():
            raise SystemExit("Exploration pareto not found; aborting adaptive pipeline")
        fine_bounds = compute_fine_search_bounds(str(pareto_lhs))
        print(f"Computed fine bounds from exploration results: {fine_bounds}")

        min_distances_list = [int(x) for x in fine_bounds]
        print("=== Phase 2: Fine Sweep (Adaptive Bounds) ===")

        from dataselector.workflows.fine_sweep import run_fine_sweep

        run_fine_sweep(
            min_distances=min_distances_list,
            output_dir=OUT / "fine_sweep",
            max_runs=fine_max_runs,
        )

    # ========================================================================
    # PHASE 3: OPTUNA OPTIMIZATION
    # ========================================================================
    pareto_fine = OUT / "fine_sweep" / "pareto_solutions.csv"
    if not pareto_fine.exists():
        raise SystemExit("Fine pareto not found; aborting adaptive pipeline")

    opt_lo, opt_hi = compute_optuna_bounds(str(pareto_fine))
    center = (opt_lo + opt_hi) // 2
    print(
        f"Optuna bounds: {opt_lo}-{opt_hi}, running Optuna with default min_distance={center}"
    )

    if skip_optuna:
        print("Skipping Optuna stage (--skip-optuna flag provided)")
    else:
        try:
            # Check if optuna is available
            try:
                import importlib.util as importlib_util

                has_optuna = importlib_util.find_spec("optuna") is not None
            except Exception:
                import importlib

                try:
                    has_optuna = importlib.find_spec("optuna") is not None
                except Exception as e_inner:
                    print(f"Warning checking for optuna using importlib: {e_inner}")
                    has_optuna = False

            if not has_optuna:
                print(
                    "Optuna not found in the current environment: skipping Optuna stage. "
                    "Install optuna to enable."
                )
            else:
                print("=== Phase 3: Optimization (Optuna) ===")

                # Decide n_samples
                if n_samples is not None:
                    chosen_n_samples = n_samples
                else:
                    try:
                        cfg_n = (
                            em.manifest.get("config", {})
                            .get("selection", {})
                            .get("n_samples")
                            if hasattr(em, "manifest")
                            else None
                        )
                    except Exception:
                        cfg_n = None

                    if cfg_n:
                        chosen_n_samples = int(cfg_n)
                    elif n_samples_min is not None and n_samples_max is not None:
                        chosen_n_samples = None  # range mode
                    else:
                        chosen_n_samples = compute_adaptive_n_initial(n_dimensions)

                print(
                    f"Running Optuna with min_distance range: {opt_lo}-{opt_hi} km "
                    f'(center {center}km); n_samples={chosen_n_samples or "range mode"}'
                )

                # Import and call Optuna optimization directly
                from dataselector.workflows.optuna_optimize import run_optuna

                # Build Optuna arguments
                constrain_bounds = {
                    "min_dist_min": int(opt_lo),
                    "min_dist_max": int(opt_hi),
                }

                n_samples_range = None
                if n_samples_min is not None and n_samples_max is not None:
                    n_samples_range = (int(n_samples_min), int(n_samples_max))

                try:
                    run_optuna(
                        n_trials=n_trials,
                        n_candidates=n_candidates,
                        n_samples=int(chosen_n_samples) if chosen_n_samples else 34,
                        n_samples_range=n_samples_range,
                        metadata_path=str(csv_path),
                        seed=seed,
                        sampler_name=optuna_sampler,
                        constrain_bounds=constrain_bounds,
                        out_dir=Path("outputs"),
                    )
                except Exception as e:
                    print("Error while running Optuna or analysis.")
                    print(repr(e))
                    if continue_on_analysis_failure:
                        print(
                            "Continuing despite Optuna failure (--continue-on-analysis-failure set)."
                        )
                    else:
                        print(
                            "Aborting pipeline due to Optuna failure. "
                            "Use --continue-on-analysis-failure to override."
                        )
                        raise SystemExit(1)

                # Analyze Optuna convergence
                trials_path = Path(em.run_dir) / "results" / "trials.csv"
                if dry_run:
                    print(
                        "Dry-run: skipping Optuna results existence check and convergence analysis"
                    )
                else:
                    if not trials_path.exists() or trials_path.stat().st_size < 10:
                        msg = f"Optuna run did not produce valid trials.csv at {trials_path}"
                        if continue_on_analysis_failure:
                            print("WARNING:", msg)
                        else:
                            print("ERROR:", msg)
                            raise SystemExit(1)

                    print(
                        "Optuna convergence analysis script has been removed in CLI hard-cut; "
                        "skipping optional convergence report generation."
                    )

        except Exception as e:
            print(
                "Warning while running Optuna or analysis. "
                "Proceeding to Bootstrap stage. Error:"
            )
            print(repr(e))

    # ========================================================================
    # PHASE 4: BOOTSTRAP VALIDATION
    # ========================================================================
    print("=== Phase 4: Validation (Bootstrap) ===")
    bootstrap_out = Path(em.run_dir) / "results" / "bootstrap_results.csv"
    bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_summary = bootstrap_out.with_name(bootstrap_out.stem + "_summary.csv")

    from dataselector.workflows.bootstrap import run_bootstrap_pareto

    run_bootstrap_pareto(
        pareto_csv=str(pareto_fine),
        n_boot=int(n_boot),
        output_csv=str(bootstrap_out),
        random_seed=int(seed),
        uq_method="bootstrap",
    )

    # Apply Bootstrap Best (optional)
    if not skip_bootstrap_injection and bootstrap_summary.exists():
        print("=== Applying Bootstrap Best ===")
        try:
            import pandas as pd
            import yaml

            summary_df = pd.read_csv(bootstrap_summary)
            if summary_df.empty:
                raise RuntimeError(f"Empty bootstrap summary: {bootstrap_summary}")

            best_row = summary_df.sort_values(
                ["jaccard_mean", "wwi_percent_mean"], ascending=False
            ).iloc[0]

            cfg_path = Path("config/pipeline_config.yaml")
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
            selection = cfg.setdefault("selection", {})
            selection["alpha_visual"] = float(best_row["alpha"])
            selection["beta_spatial"] = float(best_row["beta"])
            selection["gamma_temporal"] = float(best_row["gamma"])
            selection["min_distance_km"] = int(best_row["min_distance_km"])
            selection["n_samples"] = int(best_row["n_selected"])

            out_cfg = Path("outputs/pipeline_config.bootstrap.yaml")
            out_cfg.parent.mkdir(parents=True, exist_ok=True)
            out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(
                "✓ Bootstrap-best config written to outputs/pipeline_config.bootstrap.yaml"
            )
        except Exception as e:
            print(f"Warning: Bootstrap-best application failed: {e}")
    else:
        if skip_bootstrap_injection:
            print("Skipping Bootstrap-best injection (--skip-bootstrap-injection flag)")
        else:
            print(f"Warning: Bootstrap summary not found at {bootstrap_summary}")

    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ ADAPTIVE PIPELINE COMPLETE")
    print("=" * 80)
    print("Pipeline stages:")
    print(f"  1. {sampler.upper()} Exploration: {n_lhs} samples")
    print(f"  2. Fine Sweep: {len(fine_bounds)} adaptive bounds")
    print(f"  3. Optuna: {n_trials} trials (center={center}km)")
    print(f"  4. Bootstrap: {n_boot} resamples")
    print("=" * 80)

    # Generate experiment report
    try:
        report_dir = Path(em.run_dir) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "run_info.txt").write_text(
            f"Adaptive pipeline run completed: sampler={sampler} n_lhs={n_lhs} "
            f"at {datetime.utcnow().isoformat()}Z\n"
        )
        print(f"Generating experiment report in: {report_dir}")
        from dataselector.workflows.generate_reports import generate_experiment_report

        generate_experiment_report(report_dir)
        print(f'Report written: {report_dir / "experiment_report.md"}')
    except Exception as e:
        print(f"Warning: automatic report generation failed: {e}")

    return Path(em.run_dir)


@cli_command(
    "adaptive-pipeline",
    help="Adaptive multi-stage pipeline: Exploration → Fine → Optuna → Bootstrap",
    args={
        "experiment_name": {
            "type": str,
            "default": "adaptive_pipeline",
            "help": "Name for this experiment run",
        },
        "csv_path": {
            "type": str,
            "default": None,
            "help": "Path to tile metadata CSV (default: data/new_all_tiles.csv)",
        },
        "n_lhs": {
            "type": int,
            "default": None,
            "help": "Number of LHS samples (None = adaptive)",
        },
        "n_trials": {
            "type": int,
            "default": 100,
            "help": "Number of Optuna trials",
        },
        "n_boot": {
            "type": int,
            "default": 500,
            "help": "Number of bootstrap resamples",
        },
        "n_candidates": {
            "type": int,
            "default": None,
            "help": "Total candidate pool size (None = dataset size)",
        },
        "n_dimensions": {
            "type": int,
            "default": 9,
            "help": "Number of parameter dimensions for adaptive sizing",
        },
        "sampler": {
            "type": str,
            "default": None,
            "choices": ["lhs", "sobol"],
            "help": "Exploration sampler type (auto-selected from benchmark if available)",
        },
        "optuna_sampler": {
            "type": str,
            "default": "TPESampler",
            "help": "Optuna sampler class name",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Random seed",
        },
        "n_initial_strategy": {
            "type": str,
            "default": "conservative",
            "choices": ["conservative", "moderate", "aggressive"],
            "help": "Strategy for adaptive n_lhs computation",
        },
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Fixed n_samples for Optuna (overrides adaptive sizing)",
        },
        "n_samples_min": {
            "type": int,
            "default": None,
            "help": "Min n_samples (range mode)",
        },
        "n_samples_max": {
            "type": int,
            "default": None,
            "help": "Max n_samples (range mode)",
        },
        "fine_max_runs": {
            "type": int,
            "default": None,
            "help": "Max runs for fine sweep",
        },
        "skip_exploration": {
            "type": bool,
            "action": "store_true",
            "help": "Skip exploration phase",
        },
        "skip_fine": {
            "type": bool,
            "action": "store_true",
            "help": "Skip fine sweep",
        },
        "skip_optuna": {
            "type": bool,
            "action": "store_true",
            "help": "Skip Optuna phase",
        },
        "skip_bootstrap_injection": {
            "type": bool,
            "action": "store_true",
            "help": "Skip bootstrap-best config injection",
        },
        "continue_on_analysis_failure": {
            "type": bool,
            "action": "store_true",
            "help": "Continue pipeline on analysis errors",
        },
        "dry_run": {
            "type": bool,
            "action": "store_true",
            "help": "Dry run mode",
        },
        "pre_names": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Pre-selected tile names",
        },
        "pre_indices": {
            "type": int,
            "nargs": "*",
            "default": None,
            "help": "Pre-selected indices",
        },
        "hamburg": {
            "type": bool,
            "action": "store_true",
            "help": "Add Hamburg to pre-names",
        },
        "KDR146": {
            "type": bool,
            "action": "store_true",
            "help": "Add KDR_146 to pre-names",
        },
    },
)
def main(
    experiment_name: str = "adaptive_pipeline",
    csv_path: str | None = None,
    n_lhs: int | None = None,
    n_trials: int = 100,
    n_boot: int = 500,
    n_candidates: int | None = None,
    n_dimensions: int = 9,
    sampler: str | None = None,
    optuna_sampler: str = "TPESampler",
    seed: int = 42,
    n_initial_strategy: str = "conservative",
    n_samples: int | None = None,
    n_samples_min: int | None = None,
    n_samples_max: int | None = None,
    fine_max_runs: int | None = None,
    skip_exploration: bool = False,
    skip_fine: bool = False,
    skip_optuna: bool = False,
    skip_bootstrap_injection: bool = False,
    continue_on_analysis_failure: bool = False,
    dry_run: bool = False,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
    hamburg: bool = False,
    KDR146: bool = False,
) -> int:
    """CLI entry point for adaptive pipeline."""

    # Convert csv_path string to Path if provided
    csv_path_obj = Path(csv_path) if csv_path else None

    run_dir = run_adaptive_pipeline(
        experiment_name=experiment_name,
        csv_path=csv_path_obj,
        n_lhs=n_lhs,
        n_trials=n_trials,
        n_boot=n_boot,
        n_candidates=n_candidates,
        n_dimensions=n_dimensions,
        sampler=sampler,
        optuna_sampler=optuna_sampler,
        seed=seed,
        n_initial_strategy=n_initial_strategy,
        n_samples=n_samples,
        n_samples_min=n_samples_min,
        n_samples_max=n_samples_max,
        fine_max_runs=fine_max_runs,
        skip_exploration=skip_exploration,
        skip_fine=skip_fine,
        skip_optuna=skip_optuna,
        skip_bootstrap_injection=skip_bootstrap_injection,
        continue_on_analysis_failure=continue_on_analysis_failure,
        dry_run=dry_run,
        pre_names=pre_names,
        pre_indices=pre_indices,
        hamburg=hamburg,
        KDR146=KDR146,
    )

    print(f"\n✅ Adaptive pipeline completed. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
