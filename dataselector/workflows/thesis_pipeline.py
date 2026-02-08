"""Master pipeline for thesis optimization."""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dataselector.cli_decorators import cli_command
from dataselector.runtime import activate_repro_mode, write_run_metadata

logger = logging.getLogger(__name__)


def run_thesis_pipeline(
    n_lhs: Optional[int] = None,
    n_trials: int = 100,
    skip_exploration: bool = False,
    skip_optimization: bool = False,
    skip_validation: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
    execution_profile: str = "default",
    seed: int = 42,
) -> bool:
    """
    Run complete thesis optimization pipeline.

    Phases:
        1. Exploration (LHS-based Pareto front)
        2. Optimization (Optuna Bayesian optimization)
        3. Validation (Bootstrap robustness testing)
        4. Summary (Reports and comparison)

    Args:
        n_lhs: Number of LHS samples (if None, compute adaptive default)
        n_trials: Number of Optuna trials
        skip_exploration: Skip Phase 1
        skip_optimization: Skip Phase 2
        skip_validation: Skip Phase 3
        dry_run: Show commands without executing
        output_dir: Output directory (defaults to outputs/)

    Returns:
        True if all phases succeeded, False otherwise
    """
    runtime_state = activate_repro_mode(profile=execution_profile, seed=seed)

    # Lazy imports to avoid heavy dependencies at import time
    from dataselector.workflows.generate_reports import generate_thesis_final_report
    from dataselector.workflows.optuna_optimize import run_optuna
    from dataselector.workflows.tune_weights import run_exploration

    if output_dir is None:
        output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path("data/new_all_tiles.csv")

    # Compute adaptive n_lhs if not provided
    if n_lhs is None:
        try:
            import numpy as np
            import pandas as pd

            if metadata_path.exists():
                n_tiles = len(pd.read_csv(metadata_path))
                n_lhs = max(50, int(2 * np.sqrt(n_tiles)))
                print(f"📊 Adaptive n_lhs computed from dataset: {n_lhs}")
            else:
                n_lhs = 50
                print(f"⚠️ Metadata not found; using fallback n_lhs={n_lhs}")
        except Exception:
            n_lhs = 50
            print(f"⚠️ Could not compute adaptive n_lhs; using fallback n_lhs={n_lhs}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("🚀 THESIS OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Start: {timestamp}")
    print(f"Output Directory: {output_dir}")
    print(f"n_lhs: {n_lhs}, n_trials: {n_trials}")
    print("=" * 80)

    all_success = True

    # Phase 1: Exploration (LHS Sweep)
    if not skip_exploration:
        print("\n" + "=" * 80)
        print("PHASE 1: EXPLORATION (LHS-based Pareto-Front)")
        print("=" * 80)

        if dry_run:
            print(f"[DRY-RUN] Would run: Exploration with n_lhs={n_lhs}")
        else:
            t0 = time.time()
            try:
                print(f"Running Exploration with n_lhs={n_lhs}...")
                run_exploration(
                    n_samples=n_lhs,
                    sampler="lhs",
                    seed=seed,
                    metadata_path=metadata_path,
                    output_dir=output_dir / "tuning_weights",
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 1 erfolgreich (Dauer: {elapsed:.1f}s)")
            except Exception as e:
                print(f"❌ FEHLER in Phase 1: {e}")
                all_success = False
                if not skip_optimization and not skip_validation:
                    print("⚠️ Nachfolgende Phasen könnten fehlschlagen")
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 1: Exploration")

    # Phase 2: Optimization (Optuna)
    if not skip_optimization:
        print("\n" + "=" * 80)
        print("PHASE 2: OPTIMIZATION (Optuna Bayesian)")
        print("=" * 80)

        if dry_run:
            print(f"[DRY-RUN] Would run: Optuna with n_trials={n_trials}")
        else:
            t0 = time.time()
            try:
                print(f"Running Optuna with n_trials={n_trials}...")
                run_optuna(
                    n_trials=n_trials,
                    sampler_name="cmaes",
                    metadata_path=metadata_path,
                    seed=seed,
                    out_dir=output_dir / "optuna",
                    study_name=f"thesis_optuna_{timestamp}",
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 2 erfolgreich (Dauer: {elapsed:.1f}s)")
            except Exception as e:
                print(f"❌ FEHLER in Phase 2: {e}")
                all_success = False
                if not skip_validation:
                    print("⚠️ Validation könnte fehlschlagen")
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 2: Optimization")

    # Phase 3: Validation
    if not skip_validation:
        print("\n" + "=" * 80)
        print("PHASE 3: VALIDATION (Pareto Candidate Robustness)")
        print("=" * 80)

        if dry_run:
            print("[DRY-RUN] Would run: validation over exploration Pareto candidates")
        else:
            t0 = time.time()
            try:
                from dataselector.workflows.validation import validate_pareto_candidates

                pareto_csv = (
                    output_dir / "tuning_weights" / "pareto" / "pareto_solutions.csv"
                )
                if not pareto_csv.exists():
                    raise FileNotFoundError(
                        "Validation requires Pareto candidates at "
                        f"{pareto_csv}. Run exploration first."
                    )

                print(f"Running validation for Pareto candidates: {pareto_csv}")
                validate_pareto_candidates(
                    pareto_csv=pareto_csv,
                    output_dir=output_dir / "validation",
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 3 erfolgreich (Dauer: {elapsed:.1f}s)")
            except Exception as e:
                print(f"❌ FEHLER in Phase 3: {e}")
                all_success = False
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 3: Validation")

    # Phase 4: Summary (always run unless dry-run)
    if not dry_run:
        print("\n" + "=" * 80)
        print("PHASE 4: SUMMARY REPORT")
        print("=" * 80)

        t0 = time.time()
        try:
            print("Generating final report...")
            generate_thesis_final_report(
                output_dir=output_dir,
                timestamp=timestamp,
            )
            elapsed = time.time() - t0
            print(f"✅ Phase 4 erfolgreich (Dauer: {elapsed:.1f}s)")
        except Exception as e:
            print(f"❌ FEHLER in Phase 4: {e}")
            all_success = False

    # Final summary
    print("\n" + "=" * 80)
    if all_success:
        print("✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN")
    else:
        print("❌ PIPELINE MIT FEHLERN ABGESCHLOSSEN")
    print("=" * 80)

    try:
        write_run_metadata(
            output_dir=output_dir,
            execution_profile=execution_profile,
            seed=seed,
            command=sys.argv,
            config_path=Path("config/pipeline_config.yaml"),
            runtime_state=runtime_state,
            extra={
                "n_lhs": n_lhs,
                "n_trials": n_trials,
                "skip_exploration": skip_exploration,
                "skip_optimization": skip_optimization,
                "skip_validation": skip_validation,
                "dry_run": dry_run,
            },
        )
    except Exception as exc:
        print(f"⚠️ Could not write run metadata: {exc}")

    return all_success


@cli_command(
    "thesis-pipeline",
    help="Run complete thesis optimization pipeline (4 phases)",
    args={
        "n_lhs": {
            "type": int,
            "default": None,
            "help": "Number of LHS samples (default: adaptive)",
        },
        "n_trials": {
            "type": int,
            "default": 100,
            "help": "Number of Optuna trials",
        },
        "skip_exploration": {
            "type": bool,
            "action": "store_true",
            "help": "Skip Phase 1 (Exploration)",
        },
        "skip_optimization": {
            "type": bool,
            "action": "store_true",
            "help": "Skip Phase 2 (Optimization)",
        },
        "skip_validation": {
            "type": bool,
            "action": "store_true",
            "help": "Skip Phase 3 (Validation/Bootstrap)",
        },
        "dry_run": {
            "type": bool,
            "action": "store_true",
            "help": "Show commands without executing",
        },
        "output_dir": {
            "type": str,
            "default": None,
            "help": "Output directory (default: outputs/)",
        },
        "execution_profile": {
            "type": str,
            "default": "default",
            "choices": ["default", "thesis_repro"],
            "help": "Runtime execution profile",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Global random seed for reproducible runs",
        },
    },
)
def main(
    n_lhs: Optional[int] = None,
    n_trials: int = 100,
    skip_exploration: bool = False,
    skip_optimization: bool = False,
    skip_validation: bool = False,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    execution_profile: str = "default",
    seed: int = 42,
) -> int:
    """CLI entry point for thesis pipeline."""
    # Convert str path to Path object
    output_dir_path = Path(output_dir) if output_dir else None

    # Run the pipeline
    success = run_thesis_pipeline(
        n_lhs=n_lhs,
        n_trials=n_trials,
        skip_exploration=skip_exploration,
        skip_optimization=skip_optimization,
        skip_validation=skip_validation,
        dry_run=dry_run,
        output_dir=output_dir_path,
        execution_profile=execution_profile,
        seed=seed,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
