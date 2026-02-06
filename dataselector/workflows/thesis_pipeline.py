"""
Master-Pipeline für die Thesis-Optimierung.

Führt 4 Phasen sequenziell aus:
  1. EXPLORATION (LHS): Pareto-Front visualisieren
  2. OPTIMIZATION (Optuna): Bayesian-optimierte Parameter finden
  3. VALIDATION (Bootstrap): Robustheit der Pareto-Kandidaten testen
  4. SUMMARY: Report & Vergleich
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dataselector.cli_decorators import cli_command

logger = logging.getLogger(__name__)


def run_thesis_pipeline(
    n_lhs: Optional[int] = None,
    n_trials: int = 100,
    skip_exploration: bool = False,
    skip_optimization: bool = False,
    skip_validation: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
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
    # Lazy imports to avoid heavy dependencies at import time
    from dataselector.workflows.bootstrap import bootstrap_pareto_candidates
    from dataselector.workflows.generate_reports import generate_thesis_final_report
    from dataselector.workflows.optuna_optimize import run_optuna
    from dataselector.workflows.tune_weights import run_exploration

    if output_dir is None:
        output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute adaptive n_lhs if not provided
    if n_lhs is None:
        try:
            import numpy as np
            import pandas as pd

            metadata_path = Path("data/new_all_tiles.csv")
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
                    sampler="cmaes",
                    output_dir=output_dir / "optuna",
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

    # Phase 3: Validation (Bootstrap)
    if not skip_validation:
        print("\n" + "=" * 80)
        print("PHASE 3: VALIDATION (Bootstrap Robustness)")
        print("=" * 80)

        if dry_run:
            print("[DRY-RUN] Would run: Bootstrap validation")
        else:
            t0 = time.time()
            try:
                print("Running Bootstrap validation...")
                bootstrap_pareto_candidates(
                    n_iterations=100,
                    output_dir=output_dir / "bootstrap",
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
):
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
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run thesis optimization pipeline")
    parser.add_argument("--n-lhs", type=int, default=None, help="Number of LHS samples")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--skip-exploration", action="store_true", help="Skip Phase 1")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip Phase 2")
    parser.add_argument("--skip-validation", action="store_true", help="Skip Phase 3")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    main(
        n_lhs=args.n_lhs,
        n_trials=args.n_trials,
        skip_exploration=args.skip_exploration,
        skip_optimization=args.skip_optimization,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
