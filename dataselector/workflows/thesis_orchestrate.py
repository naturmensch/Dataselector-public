"""Scientific orchestrator for thesis pipeline (precompute -> snapshot -> run)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dataselector.cli_decorators import cli_command
from dataselector.data.io import load_metadata, load_or_extract_features
from dataselector.runtime import (
    activate_repro_mode,
    load_parameter_contract,
    validate_snapshot_against_contract,
    write_run_metadata,
)
from dataselector.runtime.parameter_snapshot import load_snapshot, validate_snapshot_file
from dataselector.workflows.leakage_audit import audit_split_leakage
from dataselector.workflows.leakage_calibration import calibrate_leakage_buffer
from dataselector.workflows.optuna_autoscale import run_optuna_autoscale_workflow
from dataselector.workflows.spatial_split_builder import build_spatial_splits
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


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be boolean-compatible (got {value!r})")


def _resolve_build_splits(build_splits: Any, execution_profile: str) -> bool:
    if isinstance(build_splits, str) and build_splits.strip().lower() == "auto":
        return execution_profile == "thesis_repro"
    return _parse_bool(build_splits, label="build_splits")


def _load_split_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Spatial split policy not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid spatial split policy format: {path}")
    return payload


def _build_leakage_safe_splits(
    *,
    out_dir: Path,
    config_path: Path,
    tile_exclusion_policy: Path,
    split_policy_path: Path,
    leakage_buffer_km: str | float,
    split_seed: int,
    cache_mode: str,
    strict_real_data: bool,
    strict_scientific: bool,
) -> dict[str, Any]:
    split_policy = _load_split_policy(split_policy_path)

    os.environ["DATASELECTOR_APPLY_TILE_EXCLUSION"] = "1"
    os.environ["DATASELECTOR_TILE_EXCLUSION_POLICY"] = str(tile_exclusion_policy)

    metadata = load_metadata(
        "data/new_all_tiles.csv",
        resolve_images=False,
        strict_metric_crs=True,
        metric_epsg=25832,
        tile_exclusion_policy=tile_exclusion_policy,
        apply_tile_exclusion=True,
    )
    features = load_or_extract_features(
        out_dir=out_dir / "parameter_resolution",
        csv_meta="data/new_all_tiles.csv",
        batch_size=16,
        cache=True,
        cache_mode=cache_mode,
        config_path=str(config_path),
        enforce_canonical=True,
        strict_cache_identity=True,
        force_extract=False,
    )
    if strict_real_data and len(features) != len(metadata):
        raise RuntimeError(
            "Feature/metadata row mismatch after tile exclusion policy: "
            f"{len(features)} vs {len(metadata)}"
        )

    buffer_value: str | float
    if isinstance(leakage_buffer_km, str):
        text = leakage_buffer_km.strip().lower()
        buffer_value = "auto" if text == "auto" else float(text)
    else:
        buffer_value = float(leakage_buffer_km)

    calibration = calibrate_leakage_buffer(
        features=features,
        metadata=metadata,
        output_dir=out_dir,
        split_policy=split_policy,
        leakage_buffer_km=buffer_value,
    )
    split_result = build_spatial_splits(
        metadata=metadata,
        output_dir=out_dir,
        split_policy=split_policy,
        tile_exclusion_policy_sha256=metadata.attrs.get("tile_exclusion_policy_sha256"),
        d_leak_km=calibration.d_leak_km,
        split_seed=int(split_seed),
    )
    split_manifest = json.loads(
        split_result.split_manifest_path.read_text(encoding="utf-8")
    )
    audit = audit_split_leakage(
        metadata=metadata,
        split_manifest=split_manifest,
        d_leak_km=calibration.d_leak_km,
        output_dir=out_dir,
    )
    if strict_scientific and audit.violations_count > 0:
        raise RuntimeError(
            "Leakage audit failed: found "
            f"{audit.violations_count} inter-split pairs with edge_distance_km < d_leak."
        )

    return {
        "tile_exclusions_applied": bool(metadata.attrs.get("tile_exclusions_applied", False)),
        "tile_exclusions_count": int(metadata.attrs.get("tile_exclusions_count", 0)),
        "tile_excluded_shortnames": list(metadata.attrs.get("tile_excluded_shortnames", [])),
        "tile_exclusion_policy_sha256": metadata.attrs.get("tile_exclusion_policy_sha256"),
        "effective_tile_count": int(metadata.attrs.get("effective_tile_count", len(metadata))),
        "source_crs": metadata.attrs.get("source_crs"),
        "metric_crs": metadata.attrs.get("metric_crs"),
        "transform_applied": bool(metadata.attrs.get("transform_applied", False)),
        "d_leak_km": float(calibration.d_leak_km),
        "distance_policy_path": str(calibration.policy_json),
        "leakage_calibration_path": str(calibration.calibration_csv),
        "split_manifest_path": str(split_result.split_manifest_path),
        "split_manifest_sha256": split_result.split_manifest_sha256,
        "leakage_audit_path": str(audit.audit_csv_path),
        "leakage_violations_count": int(audit.violations_count),
        "split_sizes": split_result.split_sizes,
        "split_component_count": int(split_result.component_count),
        "min_train_val_km": audit.min_train_val_km,
        "min_train_test_km": audit.min_train_test_km,
        "min_val_test_km": audit.min_val_test_km,
    }


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
    cache_mode: str = "read_write",
    strict_evidence_root: str = "run_dir",
    strict_real_data: bool | str = True,
    tile_exclusion_policy: str = "config/tile_exclusion_policy.yaml",
    split_policy: str = "config/spatial_split_policy.yaml",
    leakage_buffer_km: str = "auto",
    build_splits: str | bool = "auto",
    split_seed: int = 42,
    force: bool = False,
    force_override_reason: str | None = None,
) -> int:
    if force and not force_override_reason:
        raise ValueError("--force requires --force-override-reason")
    strict_real_data_flag = _parse_bool(
        strict_real_data,
        label="strict_real_data",
    )
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
    split_extra: dict[str, Any] = {}

    # Keep feature extraction, autoscale and split-building on the same candidate pool.
    tile_policy_path = Path(tile_exclusion_policy)
    if tile_policy_path.exists():
        os.environ["DATASELECTOR_APPLY_TILE_EXCLUSION"] = "1"
        os.environ["DATASELECTOR_TILE_EXCLUSION_POLICY"] = str(tile_policy_path)

    # 1) Precompute required artifacts (autoscale best + selected_n_samples).
    run_optuna_autoscale_workflow(
        n_trials=autoscale_trials or [20, 40, 80, 160],
        stages=autoscale_stages or ["50", "100", "300", "full"],
        seed=seed,
        patience=autoscale_patience,
        output_dir=str(resolution_dir),
        config_path=str(config_path),
        cache_mode=cache_mode,
        strict_real_data=strict_real_data_flag,
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
        config_path=config_path,
        cache_mode=cache_mode,
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
        run_root=out_dir,
        repo_root=Path.cwd(),
        evidence_scope=strict_evidence_root,
    )
    validation_errors = snapshot_errors + contract_errors
    if validation_errors and not force:
        joined = "\n- ".join(validation_errors)
        raise RuntimeError(f"Scientific contract validation failed:\n- {joined}")

    if _resolve_build_splits(build_splits, execution_profile):
        split_extra = _build_leakage_safe_splits(
            out_dir=out_dir,
            config_path=config_path,
            tile_exclusion_policy=tile_policy_path,
            split_policy_path=Path(split_policy),
            leakage_buffer_km=leakage_buffer_km,
            split_seed=int(split_seed),
            cache_mode=cache_mode,
            strict_real_data=strict_real_data_flag,
            strict_scientific=bool(strict_scientific),
        )

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
                    "contract_validation_scope": strict_evidence_root,
                    "contract_validation_errors": contract_errors,
                    "force_override_used": bool(force),
                    "force_override_reason": force_override_reason if force else None,
                    **split_extra,
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
        config_path=config_path,
        cache_mode=cache_mode,
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
            "contract_validation_scope": strict_evidence_root,
            "contract_validation_errors": contract_errors,
            "force_override_used": bool(force),
            "force_override_reason": force_override_reason if force else None,
            "run_after_precompute": bool(run_after_precompute),
            "strict_scientific": bool(strict_scientific),
            "run_success": bool(run_ok),
            **split_extra,
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
        "cache_mode": {
            "type": str,
            "default": "read_write",
            "choices": ["off", "read_only", "write_only", "read_write"],
        },
        "strict_evidence_root": {
            "type": str,
            "default": "run_dir",
            "choices": ["run_dir", "repo_root"],
        },
        "strict_real_data": {
            "type": str,
            "default": "true",
            "choices": ["true", "false"],
        },
        "tile_exclusion_policy": {
            "type": str,
            "default": "config/tile_exclusion_policy.yaml",
        },
        "split_policy": {
            "type": str,
            "default": "config/spatial_split_policy.yaml",
        },
        "leakage_buffer_km": {
            "type": str,
            "default": "auto",
        },
        "build_splits": {
            "type": str,
            "default": "auto",
            "choices": ["auto", "true", "false"],
        },
        "split_seed": {"type": int, "default": 42},
        "force": {"type": bool, "action": "store_true"},
        "force_override_reason": {"type": str, "default": None},
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
    cache_mode: str = "read_write",
    strict_evidence_root: str = "run_dir",
    strict_real_data: bool | str = True,
    tile_exclusion_policy: str = "config/tile_exclusion_policy.yaml",
    split_policy: str = "config/spatial_split_policy.yaml",
    leakage_buffer_km: str = "auto",
    build_splits: str | bool = "auto",
    split_seed: int = 42,
    force: bool = False,
    force_override_reason: str | None = None,
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
        cache_mode=cache_mode,
        strict_evidence_root=strict_evidence_root,
        strict_real_data=strict_real_data,
        tile_exclusion_policy=tile_exclusion_policy,
        split_policy=split_policy,
        leakage_buffer_km=leakage_buffer_km,
        build_splits=build_splits,
        split_seed=split_seed,
        force=force,
        force_override_reason=force_override_reason,
    )
