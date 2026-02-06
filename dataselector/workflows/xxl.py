"""
Modern XXL Thesis Complete Pipeline Orchestrator

Streamlined, phase-based orchestration integrating:
- Autoscale results (n_samples, optimized hyperparams)
- Sampler suite best selection
- Phases 0-5: Convergence → Optimization → Bootstrap → Finalization

Phase Structure:
  Phase 0: Pre-flight & Convergence Validation
  Phase 1-4: XXL Optimization (Hamburg + KDR100 full)
  Phase 5: Bootstrap Uncertainty Quantification
  Finalization: Reports and artifacts

Usage:
    dataselector xxl --best-sampler tpe --phase full
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dataselector.cli_decorators import cli_command


def log(level: str, msg: str):
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def read_autoscale_config(out_dir: Path) -> dict:
    """Read optimized config from Autoscale phase.

    Returns dict with keys: n_samples, alpha, beta, gamma, min_distance_km
    """
    config = {
        "n_samples": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "min_distance_km": None,
    }

    # Read n_samples
    n_samp_file = out_dir / "autoscale_selected_n_samples.txt"
    if n_samp_file.exists():
        config["n_samples"] = int(n_samp_file.read_text().strip())
        log("INFO", f"Read n_samples from autoscale: {config['n_samples']}")

    # Read full best JSON
    best_json = out_dir / "autoscale_best_latest.json"
    if best_json.exists():
        try:
            data = json.loads(best_json.read_text())
            ua = data.get("user_attrs", {})
            config["alpha"] = ua.get("alpha")
            config["beta"] = ua.get("beta")
            config["gamma"] = ua.get("gamma")
            config["min_distance_km"] = ua.get("min_distance_km")
            log(
                "INFO",
                f"Read hyperparams from autoscale: α={config['alpha']:.3f}, β={config['beta']:.3f}, γ={config['gamma']:.3f}",
            )
        except Exception as e:
            log("WARNING", f"Could not read autoscale best JSON: {e}")

    return config


def run_workflow(workflow_name: str, args: list[str], smoke: bool = False) -> int:
    """Run a dataselector workflow command.

    Args:
        workflow_name: Name of the workflow (e.g., 'optuna-optimize', 'bootstrap')
        args: Additional arguments for the workflow
        smoke: Enable smoke mode

    Returns:
        Exit code
    """
    # Most workflows now have proper argparse, so no -- separator needed
    cmd = [sys.executable, "-m", "dataselector", workflow_name] + args
    if smoke and "--smoke" not in args:
        cmd.append("--smoke")

    log("INFO", f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        log("ERROR", f"Workflow {workflow_name} failed with code {result.returncode}")

    return result.returncode


def phase_0_preflight(
    autoscale_config: dict, best_sampler: str, smoke: bool = False
) -> bool:
    """Phase 0: Pre-flight checks and convergence validation."""
    log("PHASE", "=" * 70)
    log("PHASE", "PHASE 0: PRE-FLIGHT & CONVERGENCE VALIDATION")
    log("PHASE", "=" * 70)

    # Apply smoke defaults if needed
    missing = [
        k
        for k in ("n_samples", "alpha", "beta", "gamma", "min_distance_km")
        if autoscale_config.get(k) is None
    ]
    if missing:
        if smoke:
            defaults = {
                "n_samples": 40,
                "alpha": 0.33,
                "beta": 0.33,
                "gamma": 0.34,
                "min_distance_km": 11,
            }
            for k, v in defaults.items():
                if autoscale_config.get(k) is None:
                    autoscale_config[k] = v
            log("INFO", f"Using smoke-mode defaults for: {', '.join(missing)}")
        else:
            log("ERROR", f"Missing autoscale outputs: {missing}")
            log("ERROR", "Run 'dataselector autoscale' first")
            return False

    log("SUCCESS", f"✓ Autoscale: n_samples={autoscale_config['n_samples']}")
    log("SUCCESS", f"✓ Sampler: {best_sampler}")
    log(
        "SUCCESS",
        f"✓ Hyperparams: α={autoscale_config['alpha']:.3f}, β={autoscale_config['beta']:.3f}, γ={autoscale_config['gamma']:.3f}",
    )
    log("SUCCESS", "Phase 0 complete")

    return True


def create_smoke_optuna_outputs(out_dir: Path, exp_name: str) -> bool:
    """Create synthetic Optuna outputs for smoke testing without optuna package.

    This is used when running xxl in smoke mode to avoid optuna dependency.
    """
    try:
        smoke_dir = out_dir / "smoke_outputs" / exp_name
        smoke_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic Optuna study JSON
        study_json = {
            "study_name": exp_name,
            "best_trial": {
                "trial_id": 0,
                "value": 12.5,
                "params": {
                    "alpha": 0.33,
                    "beta": 0.33,
                    "gamma": 0.34,
                    "min_distance_km": 11,
                },
            },
            "trials": [
                {"trial_id": i, "value": 12.5 - i * 0.1, "status": "COMPLETE"}
                for i in range(3)
            ],
        }

        (smoke_dir / "study.json").write_text(json.dumps(study_json, indent=2))
        log("INFO", f"Created synthetic Optuna outputs for {exp_name}")
        return True
    except Exception as e:
        log("ERROR", f"Failed to create smoke outputs: {e}")
        return False


def phase_1_optimization(
    autoscale_config: dict,
    best_sampler: str,
    n_candidates: int,
    smoke: bool = False,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    """Phase 1-4: XXL Optimization on Hamburg + KDR100."""
    log("PHASE", "=" * 70)
    log("PHASE", "PHASE 1-4: XXL OPTIMIZATION")
    log("PHASE", "=" * 70)

    n_trials = 3 if smoke else 440
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")

    # Phase 1: Baseline optimization
    log("INFO", f"Phase 1: Baseline optimization (n_trials={n_trials})")
    exp_name = f"baseline_{timestamp}"

    if smoke and output_dir:
        # In smoke mode, create synthetic outputs instead of calling optuna-optimize
        if not create_smoke_optuna_outputs(output_dir, exp_name):
            return False
    else:
        args = [
            "--n-trials",
            str(n_trials),
            "--n-candidates",
            str(n_candidates),
            "--sampler",
            best_sampler,
            "--exp-name",
            exp_name,
        ]
        if seed is not None:
            args.extend(["--seed", str(seed)])

        rc = run_workflow("optuna-optimize", args, smoke=smoke)
        if rc != 0:
            log("ERROR", "Baseline optimization failed")
            return False

    # Phase 2: Reproducibility (only in non-smoke mode)
    if not smoke:
        for s in [43, 44]:
            log("INFO", f"Phase 2: Reproducibility run (seed={s})")
            repro_args = [
                "--n-trials",
                str(n_trials),
                "--n-candidates",
                str(n_candidates),
                "--sampler",
                best_sampler,
                "--seed",
                str(s),
                "--exp-name",
                f"repro_seed{s}_{timestamp}",
            ]
            rc = run_workflow("optuna-optimize", repro_args, smoke=smoke)
            if rc != 0:
                log("WARNING", f"Repro seed={s} failed; continuing")

    log("SUCCESS", "Phases 1-4 complete")
    return True


def phase_5_bootstrap(
    run_dir: Optional[Path],
    smoke: bool = False,
    seed: Optional[int] = None,
) -> bool:
    """Phase 5: Bootstrap Uncertainty Quantification."""
    log("PHASE", "=" * 70)
    log("PHASE", "PHASE 5: BOOTSTRAP UNCERTAINTY QUANTIFICATION")
    log("PHASE", "=" * 70)

    n_boot = 10 if smoke else 500

    if run_dir is None:
        log("WARNING", "No run directory provided for bootstrap; skipping")
        return True

    log("INFO", f"Running bootstrap (n_boot={n_boot}) for run: {run_dir}")

    args = [
        "--run-dir",
        str(run_dir),
        "--n-boot",
        str(n_boot),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])

    rc = run_workflow("bootstrap", args, smoke=smoke)
    if rc != 0:
        if smoke:
            log("WARNING", "Bootstrap failed in smoke mode; continuing")
        else:
            log("ERROR", "Bootstrap failed")
            return False

    log("SUCCESS", "Phase 5 complete")
    return True


def finalization(output_dir: Path, smoke: bool = False) -> bool:
    """Final: Generate thesis artifacts and reports."""
    log("PHASE", "=" * 70)
    log("PHASE", "FINALIZATION: Thesis Artifacts & Reports")
    log("PHASE", "=" * 70)

    # Create placeholder artifacts for smoke mode
    if smoke:
        placeholders = [
            output_dir / "convergence_baseline.json",
            output_dir / "kdr100_best_selection_info.json",
            output_dir / "thesis_finalization_summary.json",
        ]
        for p in placeholders:
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"status": "placeholder", "smoke": True}))
        log("INFO", "Created smoke-mode placeholders")

    # Generate reports (if generate-reports workflow exists)
    rc = run_workflow("generate-reports", [], smoke=smoke)
    if rc != 0:
        log("WARNING", "Report generation failed or unavailable")

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "phase": "thesis_finalization",
        "status": "complete",
        "smoke": smoke,
    }

    summary_file = output_dir / "thesis_finalization_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2))

    log("SUCCESS", f"Finalization complete: {summary_file}")
    return True


@cli_command(
    "xxl",
    help="Modern XXL Thesis Pipeline Orchestrator with phase-based optimization",
    args={
        "best_sampler": {
            "type": str,
            "default": "tpe",
            "help": "Best sampler from suite (qmc/tpe/cmaes)",
        },
        "phase": {
            "type": str,
            "default": "full",
            "help": "Run only a sub-phase: finalize (bootstrap+finalization) or full (default)",
        },
        "run_dir": {
            "type": str,
            "default": None,
            "help": "Run directory to operate on (for finalize phase)",
        },
        "output_dir": {
            "type": str,
            "default": "outputs",
            "help": "Output directory for results",
        },
        "n_candidates": {
            "type": int,
            "default": 676,
            "help": "Number of candidates",
        },
        "smoke": {
            "type": bool,
            "action": "store_true",
            "help": "Run in smoke mode (reduced settings for testing)",
        },
        "seed": {
            "type": int,
            "default": None,
            "help": "Random seed for reproducibility",
        },
    },
)
def main(
    best_sampler: str = "tpe",
    phase: str = "full",
    run_dir: str | None = None,
    output_dir: str = "outputs",
    n_candidates: int = 676,
    smoke: bool = False,
    seed: int | None = None,
) -> int:
    """Entry point for XXL thesis pipeline."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # Read autoscale config
    autoscale_config = read_autoscale_config(output_dir_path)

    # Validate autoscale config for full runs
    if phase == "full" and autoscale_config["n_samples"] is None and not smoke:
        log("ERROR", "No autoscale configuration found!")
        log("ERROR", "Run 'dataselector autoscale' first")
        return 1

    try:
        if phase == "finalize":
            log("INFO", "Running finalize-only phase")
            run_dir_path = Path(run_dir) if run_dir else None

            if not phase_5_bootstrap(run_dir_path, smoke=smoke, seed=seed):
                return 1

            if not finalization(output_dir_path, smoke=smoke):
                return 1

            log("SUCCESS", "Finalize phase complete")
            return 0

        # Full pipeline
        log("START", "🚀 XXL THESIS COMPLETE PIPELINE")
        log("START", "=" * 70)

        # Phase 0: Pre-flight
        if not phase_0_preflight(autoscale_config, best_sampler, smoke=smoke):
            return 1

        print()

        # Phases 1-4: Optimization
        if not phase_1_optimization(
            autoscale_config,
            best_sampler,
            n_candidates,
            smoke=smoke,
            seed=seed,
            output_dir=output_dir_path,
        ):
            return 1

        print()

        # Phase 5: Bootstrap
        if not phase_5_bootstrap(None, smoke=smoke, seed=seed):
            return 1

        print()

        # Finalization
        if not finalization(output_dir_path, smoke=smoke):
            return 1

        print()
        log("SUCCESS", "=" * 70)
        log("SUCCESS", "✅ XXL THESIS PIPELINE COMPLETE!")
        log("SUCCESS", "=" * 70)

        return 0

    except KeyboardInterrupt:
        log("ERROR", "Pipeline interrupted by user")
        return 1
    except Exception as e:
        log("ERROR", f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
