"""Scientific orchestrator for thesis pipeline (precompute -> snapshot -> run)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataselector.cli_decorators import cli_command
from dataselector.runtime import (
    activate_repro_mode,
    load_parameter_contract,
    validate_snapshot_against_contract,
    write_run_metadata,
)
from dataselector.runtime.parameter_snapshot import load_snapshot, validate_snapshot_file
from dataselector.workflows.optuna_autoscale import run_optuna_autoscale_workflow
from dataselector.workflows.thesis_pipeline import run_thesis_pipeline


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for scientific precompute/exploration. "
            "Install torch in env 'dataselector'."
        ) from exc


def _resolve_snapshot_path(output_dir: Path) -> Path:
    stable = output_dir / "final_config.yaml"
    if stable.exists():
        return stable
    candidates = sorted(output_dir.glob("final_config_*.yaml"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No resolved snapshot found in {output_dir}. "
        "Expected final_config.yaml or final_config_<timestamp>.yaml"
    )


def run_thesis_orchestrate(
    *,
    config: str = "config/pipeline_config.yaml",
    output_dir: str | None = None,
    execution_profile: str = "thesis_repro",
    seed: int = 42,
    n_samples: int | None = None,
    n_trials: int = 100,
    validation_seeds: list[int] | None = None,
    validation_min_distances: list[float] | None = None,
    autoscale_trials: list[int] | None = None,
    autoscale_stages: list[str] | None = None,
    autoscale_patience: int = 2,
    precompute_only: bool = False,
    run_after_precompute: bool = True,
    strict_scientific: bool = True,
    force: bool = False,
) -> int:
    runtime_state = activate_repro_mode(profile=execution_profile, seed=seed)
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    metadata_path = Path("data/new_all_tiles.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Canonical metadata missing: data/new_all_tiles.csv"
        )

    _require_torch()

    if output_dir:
        out_dir = Path(output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("outputs") / "runs" / f"thesis_orchestrated_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolution_dir = out_dir / "parameter_resolution"
    resolution_dir.mkdir(parents=True, exist_ok=True)

    # 1) Precompute required artifacts (autoscale best + selected_n_samples).
    run_optuna_autoscale_workflow(
        n_trials=autoscale_trials or [20, 40, 80, 160],
        stages=autoscale_stages or ["50", "100", "300", "full"],
        seed=seed,
        patience=autoscale_patience,
        output_dir=str(resolution_dir),
    )

    # 2) Resolver + snapshot stage only.
    resolution_ok = run_thesis_pipeline(
        n_trials=n_trials,
        n_samples=n_samples,
        compute_params=True,
        snapshot_config=True,
        no_auto_continue=True,
        force=force,
        output_dir=out_dir,
        execution_profile=execution_profile,
        seed=seed,
        validation_seeds=validation_seeds,
        validation_min_distances=validation_min_distances,
        strict_scientific=strict_scientific,
    )
    if not resolution_ok:
        raise RuntimeError("Resolver/snapshot phase failed.")

    snapshot_path = _resolve_snapshot_path(out_dir)
    snapshot_errors = validate_snapshot_file(snapshot_path)
    snapshot = load_snapshot(snapshot_path)
    contract = load_parameter_contract(Path("config/parameter_resolution_contract.yaml"))
    contract_errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        repo_root=Path.cwd(),
    )
    validation_errors = snapshot_errors + contract_errors
    if validation_errors and not force:
        joined = "\n- ".join(validation_errors)
        raise RuntimeError(f"Scientific contract validation failed:\n- {joined}")

    if precompute_only or not run_after_precompute:
        write_run_metadata(
            output_dir=out_dir,
            execution_profile=execution_profile,
            seed=seed,
            config_path=config_path,
            runtime_state=runtime_state,
            extra={
                "orchestrator_mode": "precompute_only",
                "snapshot_path": str(snapshot_path),
                "validation_errors": validation_errors,
                "force_override_used": bool(force),
            },
        )
        return 0

    # 3) Production thesis run from validated snapshot.
    run_ok = run_thesis_pipeline(
        n_trials=n_trials,
        n_samples=n_samples,
        compute_params=False,
        use_params=snapshot_path,
        snapshot_config=False,
        force=force,
        output_dir=out_dir,
        execution_profile=execution_profile,
        seed=seed,
        validation_seeds=validation_seeds,
        validation_min_distances=validation_min_distances,
        strict_scientific=strict_scientific,
    )

    write_run_metadata(
        output_dir=out_dir,
        execution_profile=execution_profile,
        seed=seed,
        config_path=config_path,
        runtime_state=runtime_state,
        extra={
            "orchestrator_mode": "full",
            "snapshot_path": str(snapshot_path),
            "snapshot_validated": len(validation_errors) == 0,
            "snapshot_validation_errors": validation_errors,
            "force_override_used": bool(force),
            "run_after_precompute": bool(run_after_precompute),
            "strict_scientific": bool(strict_scientific),
            "run_success": bool(run_ok),
        },
    )
    return 0 if run_ok else 1


@cli_command(
    "thesis-orchestrate",
    help="Scientific trigger-all orchestration for thesis pipeline",
    args={
        "config": {"type": str, "default": "config/pipeline_config.yaml"},
        "output_dir": {"type": str, "default": None},
        "execution_profile": {
            "type": str,
            "default": "thesis_repro",
            "choices": ["default", "thesis_repro"],
        },
        "seed": {"type": int, "default": 42},
        "n_samples": {"type": int, "default": None},
        "n_trials": {"type": int, "default": 100},
        "validation_seeds": {"type": int, "nargs": "+", "default": None},
        "validation_min_distances": {"type": float, "nargs": "+", "default": None},
        "autoscale_trials": {"type": int, "nargs": "+", "default": None},
        "autoscale_stages": {"type": str, "nargs": "+", "default": None},
        "autoscale_patience": {"type": int, "default": 2},
        "precompute_only": {"type": bool, "action": "store_true"},
        "run_after_precompute": {"type": bool, "default": True},
        "strict_scientific": {"type": bool, "default": True},
        "force": {"type": bool, "action": "store_true"},
    },
)
def cli_thesis_orchestrate(
    config: str = "config/pipeline_config.yaml",
    output_dir: str | None = None,
    execution_profile: str = "thesis_repro",
    seed: int = 42,
    n_samples: int | None = None,
    n_trials: int = 100,
    validation_seeds: list[int] | None = None,
    validation_min_distances: list[float] | None = None,
    autoscale_trials: list[int] | None = None,
    autoscale_stages: list[str] | None = None,
    autoscale_patience: int = 2,
    precompute_only: bool = False,
    run_after_precompute: bool = True,
    strict_scientific: bool = True,
    force: bool = False,
) -> int:
    return run_thesis_orchestrate(
        config=config,
        output_dir=output_dir,
        execution_profile=execution_profile,
        seed=seed,
        n_samples=n_samples,
        n_trials=n_trials,
        validation_seeds=validation_seeds,
        validation_min_distances=validation_min_distances,
        autoscale_trials=autoscale_trials,
        autoscale_stages=autoscale_stages,
        autoscale_patience=autoscale_patience,
        precompute_only=precompute_only,
        run_after_precompute=run_after_precompute,
        strict_scientific=strict_scientific,
        force=force,
    )

