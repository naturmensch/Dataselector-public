"""Adaptive-auto orchestrator.

Composes existing workflows:
1) Autoscale (optional, when n_samples not provided)
2) Adaptive pipeline (always)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataselector.cli_decorators import cli_command
from dataselector.runtime import activate_repro_mode, write_run_metadata
from dataselector.workflows.adaptive_pipeline import main as adaptive_pipeline_main
from dataselector.workflows.autoscale import main as autoscale_main


def _resolve_n_samples_from_autoscale(output_dir: Path) -> Optional[int]:
    selected = output_dir / "autoscale_selected_n_samples.txt"
    if not selected.exists():
        return None
    try:
        return int(selected.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def run_adaptive_auto(
    csv: str,
    output_dir: str = "outputs",
    experiment_name: str = "adaptive_auto",
    n_samples: int | None = None,
    n_trials: int = 100,
    n_boot: int = 500,
    n_candidates: int | None = None,
    sampler: str = "lhs",
    optuna_sampler: str = "TPESampler",
    seed: int = 42,
    n_lhs: int | None = None,
    n_initial_strategy: str = "modern",
    autoscale_trials: list[int] | None = None,
    autoscale_stages: list[str] | None = None,
    autoscale_patience: int = 2,
    execution_profile: str = "default",
    dry_run: bool = False,
) -> int:
    """Run adaptive-auto flow with explicit autoscale handoff when needed."""
    runtime_state = activate_repro_mode(profile=execution_profile, seed=seed)
    used_autoscale = n_samples is None
    csv_path = Path(csv)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if dry_run:
        if n_samples is None:
            print(
                "adaptive-auto dry-run: would run autoscale to determine n_samples, "
                "then execute adaptive-pipeline."
            )
        else:
            print(
                f"adaptive-auto dry-run: would execute adaptive-pipeline with n_samples={n_samples}."
            )
        write_run_metadata(
            output_dir=out_root,
            execution_profile=execution_profile,
            seed=seed,
            runtime_state=runtime_state,
            extra={
                "csv": str(csv_path),
                "experiment_name": experiment_name,
                "dry_run": True,
            },
        )
        return 0

    if n_samples is None:
        if autoscale_trials is None:
            autoscale_trials = [20, 40, 80, 160]
        if autoscale_stages is None:
            autoscale_stages = ["50", "100", "300", "full"]

        print(
            "adaptive-auto: n_samples not provided; running autoscale to determine "
            "best n_samples."
        )
        rc = autoscale_main(
            csv=str(csv_path),
            n_trials=autoscale_trials,
            stages=autoscale_stages,
            output_dir=str(out_root),
            n_candidates=n_candidates,
            seed=seed,
            patience=autoscale_patience,
            pre_names=None,
            pre_indices=None,
        )
        if rc != 0:
            print(f"adaptive-auto: autoscale failed with exit code {rc}")
            return rc

        n_samples = _resolve_n_samples_from_autoscale(out_root)
        if n_samples is None:
            print(
                "adaptive-auto: autoscale completed but no autoscale_selected_n_samples.txt "
                "was produced."
            )
            return 1

    print(f"adaptive-auto: running adaptive-pipeline with n_samples={n_samples}")
    rc = adaptive_pipeline_main(
        experiment_name=experiment_name,
        csv_path=str(csv_path),
        n_lhs=n_lhs,
        n_trials=n_trials,
        n_boot=n_boot,
        n_candidates=n_candidates,
        sampler=sampler,
        optuna_sampler=optuna_sampler,
        seed=seed,
        n_initial_strategy=n_initial_strategy,
        n_samples=n_samples,
        dry_run=dry_run,
    )
    write_run_metadata(
        output_dir=out_root,
        execution_profile=execution_profile,
        seed=seed,
        runtime_state=runtime_state,
        extra={
            "csv": str(csv_path),
            "experiment_name": experiment_name,
            "n_samples": n_samples,
            "sampler": sampler,
            "optuna_sampler": optuna_sampler,
            "autoscale_used": used_autoscale,
            "exit_code": rc,
        },
    )
    return rc


@cli_command(
    "adaptive-auto",
    help="Adaptive orchestrator: autoscale n_samples (if needed) + adaptive-pipeline",
    args={
        "csv": {
            "type": str,
            "required": True,
            "help": "Path to tile metadata CSV",
        },
        "output_dir": {
            "type": str,
            "default": "outputs",
            "help": "Output directory for autoscale artifacts",
        },
        "experiment_name": {
            "type": str,
            "default": "adaptive_auto",
            "help": "Experiment name for adaptive-pipeline run",
        },
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Fixed n_samples (if omitted, autoscale chooses it)",
        },
        "n_trials": {
            "type": int,
            "default": 100,
            "help": "Optuna trials for adaptive-pipeline",
        },
        "n_boot": {
            "type": int,
            "default": 500,
            "help": "Bootstrap resamples for adaptive-pipeline",
        },
        "n_candidates": {
            "type": int,
            "default": None,
            "help": "Candidate pool size override",
        },
        "sampler": {
            "type": str,
            "choices": ["lhs", "sobol"],
            "default": "lhs",
            "help": "Exploration sampler for adaptive-pipeline",
        },
        "optuna_sampler": {
            "type": str,
            "default": "TPESampler",
            "help": "Optuna sampler class name (e.g. TPESampler, QMCSampler)",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Random seed",
        },
        "n_lhs": {
            "type": int,
            "default": None,
            "help": "Exploration sample size override",
        },
        "n_initial_strategy": {
            "type": str,
            "choices": ["modern", "legacy"],
            "default": "modern",
            "help": "Adaptive strategy for n_lhs",
        },
        "autoscale_trials": {
            "type": int,
            "nargs": "+",
            "default": None,
            "help": "Autoscale trial counts per stage when n_samples is not provided",
        },
        "autoscale_stages": {
            "type": str,
            "nargs": "+",
            "default": None,
            "help": "Autoscale stage sample sizes (e.g. 50 100 300 full)",
        },
        "autoscale_patience": {
            "type": int,
            "default": 2,
            "help": "Autoscale early-stopping patience",
        },
        "execution_profile": {
            "type": str,
            "choices": ["default", "thesis_repro"],
            "default": "default",
            "help": "Runtime execution profile",
        },
        "dry_run": {
            "type": bool,
            "action": "store_true",
            "help": "Dry-run adaptive-pipeline stage",
        },
    },
)
def main(
    csv: str,
    output_dir: str = "outputs",
    experiment_name: str = "adaptive_auto",
    n_samples: int | None = None,
    n_trials: int = 100,
    n_boot: int = 500,
    n_candidates: int | None = None,
    sampler: str = "lhs",
    optuna_sampler: str = "TPESampler",
    seed: int = 42,
    n_lhs: int | None = None,
    n_initial_strategy: str = "modern",
    autoscale_trials: list[int] | None = None,
    autoscale_stages: list[str] | None = None,
    autoscale_patience: int = 2,
    execution_profile: str = "default",
    dry_run: bool = False,
) -> int:
    return run_adaptive_auto(
        csv=csv,
        output_dir=output_dir,
        experiment_name=experiment_name,
        n_samples=n_samples,
        n_trials=n_trials,
        n_boot=n_boot,
        n_candidates=n_candidates,
        sampler=sampler,
        optuna_sampler=optuna_sampler,
        seed=seed,
        n_lhs=n_lhs,
        n_initial_strategy=n_initial_strategy,
        autoscale_trials=autoscale_trials,
        autoscale_stages=autoscale_stages,
        autoscale_patience=autoscale_patience,
        execution_profile=execution_profile,
        dry_run=dry_run,
    )
