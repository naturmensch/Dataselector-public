"""Master pipeline for thesis optimization."""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dataselector.cli_decorators import cli_command
from dataselector.runtime import activate_repro_mode, write_run_metadata
from dataselector.runtime.parameter_snapshot import (
    build_snapshot,
    compute_file_sha256,
    load_snapshot,
    validate_snapshot_file,
    write_snapshot,
)
from dataselector.workflows._selection_target import resolve_selection_n_samples

logger = logging.getLogger(__name__)


def _config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Return nested config value via dotted path."""
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _config_first(config: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = _config_get(config, path, default=None)
        if value is not None:
            return value
    return None


def _ensure_provenance_section(parameters: dict[str, Any], section: str) -> dict[str, Any]:
    sec = parameters.setdefault(section, {})
    if not isinstance(sec, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    prov = sec.setdefault("_provenance", {})
    if not isinstance(prov, dict):
        raise ValueError(f"Config section '{section}._provenance' must be a mapping")
    return prov


def _record_param_provenance(
    parameters: dict[str, Any],
    section: str,
    key: str,
    *,
    method: str,
    source_file: str | None = None,
    source_hash: str | None = None,
    compute_args: dict[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    """Attach additive provenance metadata under <section>._provenance.<key>."""
    prov = _ensure_provenance_section(parameters, section)
    entry: dict[str, Any] = {
        "method": method,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if source_file:
        entry["source_file"] = source_file
    if source_hash:
        entry["source_hash"] = source_hash
    if compute_args:
        entry["compute_args"] = compute_args
    if notes:
        entry["notes"] = notes
    prov[key] = entry


def _parse_positive_int(value: Any, *, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be > 0 (got {parsed})")
    return parsed


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be boolean-compatible (got {value!r})")


def _read_selected_sampler(path: Path) -> str | None:
    """Read sampler artifact JSON and return selected sampler name."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    best = payload.get("best") or payload.get("sampler")
    if best is None:
        return None
    sampler = str(best).strip().lower()
    if sampler in {"qmc", "tpe", "cmaes"}:
        return sampler
    return None


def _resolve_optuna_sampler(
    *,
    config: dict[str, Any],
    output_dir: Path,
    n_trials: int,
    n_samples: int,
    validation_seeds: list[int],
    compute_params: bool,
    dry_run: bool,
) -> tuple[str | None, str, str | None]:
    """Resolve Optuna sampler by policy > artifact > auto-compare."""
    sampler_policy = _config_first(
        config,
        [
            "selection.optuna_sampler",
            "selection.sampler_policy",
            "selection.sampler",
            "optimization.sampler",
        ],
    )
    if sampler_policy is not None:
        sampler = str(sampler_policy).strip().lower()
        if sampler not in {"qmc", "tpe", "cmaes"}:
            raise ValueError(
                "Invalid configured sampler policy '{}'. Expected one of qmc|tpe|cmaes.".format(
                    sampler_policy
                )
            )
        return sampler, "config_policy", None

    artifact_candidates = [
        output_dir / "selected_sampler.json",
        output_dir / "sampler_resolution" / "selected_sampler.json",
        Path("outputs") / "selected_sampler.json",
    ]
    for candidate in artifact_candidates:
        sampler = _read_selected_sampler(candidate)
        if sampler:
            return sampler, f"artifact:{candidate}", str(candidate)

    if compute_params:
        if dry_run:
            # Dry-runs do not execute compare-samplers.
            return None, "auto_compare_dry_run", None

        from dataselector.workflows.compare_samplers import compare_multi_seed

        compare_out = output_dir / "sampler_resolution"
        compare_multi_seed(
            samplers=["qmc", "tpe", "cmaes"],
            seeds=validation_seeds,
            n_trials=max(1, int(n_trials)),
            datasets=["kdr100"],
            output=str(compare_out),
            n_samples=int(n_samples),
        )
        artifact = compare_out / "selected_sampler.json"
        sampler = _read_selected_sampler(artifact)
        if sampler:
            return sampler, "auto_compare", str(artifact)

    return None, "unresolved", None


def _resolve_exploration_sampler(
    *,
    config: dict[str, Any],
    resolved_optuna_sampler: str | None,
) -> tuple[str | None, str]:
    """Resolve exploration sampler without hardcoded production literals.

    Priority:
        1. explicit exploration policy in config/snapshot
        2. deterministic mapping from resolved optuna sampler
        3. unresolved
    """
    sampler_policy = _config_first(
        config,
        [
            "selection.exploration_sampler",
            "selection.exploration.sampler",
            "selection.sampler_exploration",
            "selection.exploration_policy",
        ],
    )
    if sampler_policy is not None:
        sampler = str(sampler_policy).strip().lower()
        if sampler not in {"lhs", "sobol"}:
            raise ValueError(
                "Invalid exploration sampler policy '{}'. Expected lhs|sobol.".format(
                    sampler_policy
                )
            )
        return sampler, "config_policy"

    if resolved_optuna_sampler is not None:
        from dataselector.workflows.thesis_sampler_suite import (
            map_sampler_for_adaptive_pipeline,
        )

        explore_sampler, _ = map_sampler_for_adaptive_pipeline(resolved_optuna_sampler)
        sampler = str(explore_sampler).strip().lower()
        if sampler in {"lhs", "sobol"}:
            return sampler, "mapped_from_optuna_sampler"

    return None, "unresolved"


def _read_autoscale_best_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_selection_values_from_autoscale_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    params = payload.get("params")
    user_attrs = payload.get("user_attrs")
    if not isinstance(params, dict):
        params = {}
    if not isinstance(user_attrs, dict):
        user_attrs = {}

    out: dict[str, Any] = {}

    # Prefer explicitly persisted normalized values.
    alpha = user_attrs.get("alpha")
    beta = user_attrs.get("beta")
    gamma = user_attrs.get("gamma")
    if alpha is None or beta is None or gamma is None:
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is not None and b is not None and c is not None:
            a = float(a)
            b = float(b)
            c = float(c)
            total = a + b + c
            if total > 0:
                alpha = a / total
                beta = b / total
                gamma = c / total

    if alpha is not None:
        out["alpha_visual"] = float(alpha)
    if beta is not None:
        out["beta_spatial"] = float(beta)
    if gamma is not None:
        out["gamma_temporal"] = float(gamma)

    min_distance_km = user_attrs.get("min_distance_km", params.get("min_distance_km"))
    if min_distance_km is not None:
        out["min_distance_km"] = float(min_distance_km)

    n_samples = user_attrs.get("n_samples")
    if n_samples is not None:
        out["n_samples"] = int(n_samples)

    return out


def _resolve_computed_selection_values(
    *,
    compute_params: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], str | None, str | None, str | None]:
    """Resolve computed selection values from canonical autoscale artifacts.

    Returns:
        (values, method, source_file, source_hash)
    """
    if not compute_params:
        return {}, None, None, None

    artifact_candidates = [
        output_dir / "parameter_resolution" / "optuna_autoscale_best_latest.json",
        output_dir / "optuna_autoscale_best_latest.json",
        output_dir / "parameter_resolution" / "autoscale_best_latest.json",
        output_dir / "autoscale_best_latest.json",
        Path("outputs") / "optuna_autoscale_best_latest.json",
        Path("outputs") / "autoscale_best_latest.json",
    ]

    for candidate in artifact_candidates:
        payload = _read_autoscale_best_payload(candidate)
        if payload is None:
            continue
        values = _extract_selection_values_from_autoscale_payload(payload)
        if values:
            return (
                values,
                "computed_autoscale_artifact",
                str(candidate),
                compute_file_sha256(candidate),
            )

    return {}, None, None, None


def run_thesis_pipeline(
    n_lhs: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_trials: int = 100,
    compute_params: bool = False,
    use_params: Optional[Path] = None,
    snapshot_config: bool = False,
    no_auto_continue: bool = False,
    force: bool = False,
    skip_exploration: bool = False,
    skip_optimization: bool = False,
    skip_validation: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
    execution_profile: str = "default",
    seed: int = 42,
    pre_names: Optional[list[str]] = None,
    pre_indices: Optional[list[int]] = None,
    hamburg: bool = False,
    validation_seeds: Optional[list[int]] = None,
    validation_min_distances: Optional[list[float]] = None,
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
        n_samples: Target number of selected tiles (resolved if None)
        n_trials: Number of Optuna trials
        skip_exploration: Skip Phase 1
        skip_optimization: Skip Phase 2
        skip_validation: Skip Phase 3
        dry_run: Show commands without executing
        output_dir: Output directory (defaults to outputs/)
        pre_names: Optional pre-selected tile names
        pre_indices: Optional pre-selected tile indices
        hamburg: Convenience flag to add "Hamburg" to pre-selected names
        validation_seeds: Optional validation seed list (quick gate default: [seed])
        validation_min_distances: Optional min_distance list for validation

    Returns:
        True if all phases succeeded, False otherwise
    """
    runtime_state = activate_repro_mode(profile=execution_profile, seed=seed)

    # Lazy imports to avoid heavy dependencies at import time
    from dataselector.workflows.generate_reports import generate_thesis_final_report
    from dataselector.workflows.optuna_optimize import run_optuna
    from dataselector.workflows.tune_weights import run_exploration

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = Path("outputs") / "runs" / f"thesis_pipeline_{run_timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path("data/new_all_tiles.csv")
    config_path = Path("config/pipeline_config.yaml")

    snapshot_errors: list[str] = []
    snapshot_forced = False
    snapshot_input_path: Path | None = None
    sampler_artifact_path: str | None = None

    # Resolve active parameters from canonical config or validated snapshot.
    if use_params is not None:
        snapshot_input_path = Path(use_params)
        if not snapshot_input_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_input_path}")
        snapshot_errors = validate_snapshot_file(snapshot_input_path)
        if snapshot_errors and not force:
            raise ValueError(
                "Snapshot validation failed:\n- " + "\n- ".join(snapshot_errors)
            )
        if snapshot_errors and force:
            snapshot_forced = True
            print(
                "⚠️ Snapshot validation failed but --force enabled; continuing.\n- "
                + "\n- ".join(snapshot_errors)
            )
        snapshot_payload = load_snapshot(snapshot_input_path)
        parameters = snapshot_payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Snapshot {snapshot_input_path} missing top-level 'parameters' mapping"
            )
        parameter_source = f"snapshot:{snapshot_input_path}"
    else:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            parameters = yaml.safe_load(f) or {}
        if not isinstance(parameters, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        parameter_source = f"config:{config_path}"

    policy_source_path = snapshot_input_path if snapshot_input_path else config_path
    policy_source_file = str(policy_source_path)
    policy_source_hash = (
        compute_file_sha256(policy_source_path)
        if policy_source_path.exists()
        else None
    )
    policy_method = "snapshot_policy" if snapshot_input_path is not None else "config_policy"

    computed_selection_values, computed_selection_method, computed_selection_source_file, computed_selection_source_hash = _resolve_computed_selection_values(
        compute_params=compute_params,
        output_dir=output_dir,
    )

    sentinel = object()

    def resolve_param(
        *,
        section: str,
        key: str,
        paths: list[str],
        parser,
        default: Any = sentinel,
        compute_fn=None,
        compute_method: str | None = None,
        computed_value: Any = sentinel,
        computed_method: str | None = None,
        computed_source_file: str | None = None,
        computed_source_hash: str | None = None,
        prefer_computed_when_requested: bool = False,
        notes: str | None = None,
    ) -> Any:
        raw_value = _config_first(parameters, paths)
        method = policy_method
        source_file = policy_source_file
        source_hash = policy_source_hash
        compute_args: dict[str, Any] = {"paths": paths}

        if (
            compute_params
            and computed_value is not sentinel
            and (prefer_computed_when_requested or raw_value is None)
        ):
            value = parser(computed_value)
            method = computed_method or "computed"
            source_file = computed_source_file
            source_hash = computed_source_hash
            compute_args["computed_from_artifact"] = True
        elif raw_value is None:
            if compute_fn is not None and compute_params:
                value = parser(compute_fn())
                method = compute_method or "computed"
                source_file = None
                source_hash = None
                compute_args["computed"] = True
            elif default is not sentinel:
                value = parser(default)
                method = "workflow_default_policy"
                source_file = None
                source_hash = None
                compute_args["default"] = default
                print(
                    f"⚠️ Using explicit workflow default for {section}.{key}: {value!r}"
                )
            else:
                raise ValueError(
                    f"{section}.{key} unresolved. Set config value"
                    + (" or use --compute-params." if compute_fn is not None else ".")
                )
        else:
            value = parser(raw_value)

        parameters.setdefault(section, {})[key] = value
        _record_param_provenance(
            parameters,
            section,
            key,
            method=method,
            source_file=source_file,
            source_hash=source_hash,
            compute_args=compute_args,
            notes=notes,
        )
        return value

    # Resolve selection target without silent numeric fallback.
    if n_samples is not None:
        resolved_n_samples = _parse_positive_int(n_samples, label="n_samples")
        n_samples_source = "explicit"
    elif compute_params and computed_selection_values.get("n_samples") is not None:
        resolved_n_samples = _parse_positive_int(
            computed_selection_values["n_samples"], label="selection.n_samples"
        )
        n_samples_source = (
            computed_selection_method
            if computed_selection_method is not None
            else "computed"
        )
    else:
        cfg_n_samples = _config_get(parameters, "selection.n_samples")
        if cfg_n_samples is not None:
            resolved_n_samples = _parse_positive_int(
                cfg_n_samples, label="selection.n_samples"
            )
            n_samples_source = (
                "snapshot_config" if snapshot_input_path is not None else "config"
            )
        else:
            resolved_n_samples, n_samples_source = resolve_selection_n_samples(
                None,
                context="thesis_pipeline",
                root=Path.cwd(),
                config_path=config_path,
                experiment_run_dir=output_dir,
            )
    parameters.setdefault("selection", {})["n_samples"] = int(resolved_n_samples)
    _record_param_provenance(
        parameters,
        "selection",
        "n_samples",
        method=f"resolved:{n_samples_source}",
        source_file=(
            policy_source_file
            if n_samples_source in {"config", "snapshot_config"}
            else computed_selection_source_file
            if n_samples_source == "computed_autoscale_artifact"
            else None
        ),
        source_hash=(
            policy_source_hash
            if n_samples_source in {"config", "snapshot_config"}
            else computed_selection_source_hash
            if n_samples_source == "computed_autoscale_artifact"
            else None
        ),
        compute_args={"explicit_cli": n_samples is not None},
    )

    # Resolve and record critical policy/config parameters centrally.
    resolve_param(
        section="selection",
        key="metric",
        paths=["selection.metric"],
        parser=lambda v: str(v).strip(),
    )
    resolve_param(
        section="selection",
        key="alpha_visual",
        paths=["selection.alpha_visual", "selection.weights.alpha"],
        parser=float,
        computed_value=computed_selection_values.get("alpha_visual", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
    )
    resolve_param(
        section="selection",
        key="beta_spatial",
        paths=["selection.beta_spatial", "selection.weights.beta"],
        parser=float,
        computed_value=computed_selection_values.get("beta_spatial", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
    )
    resolve_param(
        section="selection",
        key="gamma_temporal",
        paths=["selection.gamma_temporal", "selection.weights.gamma"],
        parser=float,
        computed_value=computed_selection_values.get("gamma_temporal", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
    )
    resolve_param(
        section="selection",
        key="spatial_constraint",
        paths=["selection.spatial_constraint"],
        parser=lambda v: _parse_bool(v, label="selection.spatial_constraint"),
    )
    resolve_param(
        section="selection",
        key="use_multi_criteria",
        paths=["selection.use_multi_criteria"],
        parser=lambda v: _parse_bool(v, label="selection.use_multi_criteria"),
    )
    resolve_param(
        section="selection",
        key="use_constraint_integration",
        paths=["selection.use_constraint_integration"],
        parser=lambda v: _parse_bool(v, label="selection.use_constraint_integration"),
    )
    resolve_param(
        section="selection",
        key="random_state",
        paths=["selection.random_state"],
        parser=int,
    )

    resolved_n_clusters = resolve_param(
        section="clustering",
        key="n_clusters",
        paths=["clustering.n_clusters"],
        parser=int,
    )
    resolved_umap_components = resolve_param(
        section="clustering",
        key="umap_components",
        paths=["clustering.umap_components"],
        parser=int,
    )
    resolved_umap_n_neighbors = resolve_param(
        section="clustering",
        key="umap_n_neighbors",
        paths=["clustering.umap_n_neighbors"],
        parser=int,
    )
    resolved_umap_random_state = resolve_param(
        section="clustering",
        key="umap_random_state",
        paths=["clustering.umap_random_state"],
        parser=int,
    )
    resolved_umap_n_jobs = resolve_param(
        section="clustering",
        key="umap_n_jobs",
        paths=["clustering.umap_n_jobs"],
        parser=int,
    )

    resolve_param(
        section="feature_extraction",
        key="model",
        paths=["feature_extraction.model"],
        parser=lambda v: str(v).strip().lower(),
    )
    resolved_batch_size = resolve_param(
        section="feature_extraction",
        key="batch_size",
        paths=["feature_extraction.batch_size", "data.batch_size"],
        parser=int,
    )
    crop_size_value = resolve_param(
        section="feature_extraction",
        key="crop_size",
        paths=["feature_extraction.crop_size"],
        parser=lambda v: [int(v[0]), int(v[1])] if isinstance(v, (list, tuple)) and len(v) == 2 else (_parse_positive_int(v, label="feature_extraction.crop_size"), _parse_positive_int(v, label="feature_extraction.crop_size")),
    )
    if isinstance(crop_size_value, tuple):
        crop_size_value = [int(crop_size_value[0]), int(crop_size_value[1])]
        parameters.setdefault("feature_extraction", {})["crop_size"] = crop_size_value
    resolve_param(
        section="feature_extraction",
        key="device",
        paths=["feature_extraction.device"],
        parser=lambda v: str(v).strip().lower(),
    )

    # Resolve min_distance policy from config/snapshot or compute (if requested).
    computed_min_distance_km = computed_selection_values.get("min_distance_km", sentinel)
    config_min_distance_km = _config_get(parameters, "selection.min_distance_km")
    if compute_params and computed_min_distance_km is not sentinel:
        config_min_distance_km = float(computed_min_distance_km)
        parameters.setdefault("selection", {})["min_distance_km"] = config_min_distance_km
        _record_param_provenance(
            parameters,
            "selection",
            "min_distance_km",
            method=computed_selection_method or "computed",
            source_file=computed_selection_source_file,
            source_hash=computed_selection_source_hash,
            compute_args={"source": "autoscale_artifact"},
        )
    elif config_min_distance_km is not None:
        config_min_distance_km = float(config_min_distance_km)
        _record_param_provenance(
            parameters,
            "selection",
            "min_distance_km",
            method=policy_method,
            source_file=policy_source_file,
            source_hash=policy_source_hash,
        )
    elif compute_params:
        from dataselector.pipeline.pipeline_utils import compute_min_distance_km

        config_min_distance_km = float(compute_min_distance_km(str(metadata_path)))
        parameters.setdefault("selection", {})["min_distance_km"] = config_min_distance_km
        _record_param_provenance(
            parameters,
            "selection",
            "min_distance_km",
            method="computed_from_metadata",
            source_file=str(metadata_path),
            source_hash=compute_file_sha256(metadata_path)
            if metadata_path.exists()
            else None,
            compute_args={"function": "compute_min_distance_km"},
        )
    else:
        raise ValueError(
            "selection.min_distance_km unresolved. Set config value or use --compute-params."
        )

    # Validation seeds resolve from CLI > config > explicit default.
    if validation_seeds is None:
        cfg_seeds = _config_first(
            parameters,
            ["selection.validation_seeds", "validation.seeds"],
        )
        if cfg_seeds is not None:
            if not isinstance(cfg_seeds, (list, tuple)):
                raise ValueError(
                    f"selection.validation_seeds must be a list (got {type(cfg_seeds)})"
                )
            validation_seeds = [int(s) for s in cfg_seeds]
            _record_param_provenance(
                parameters,
                "selection",
                "validation_seeds",
                method=policy_method,
                source_file=policy_source_file,
                source_hash=policy_source_hash,
                compute_args={"source": "config"},
            )
        else:
            validation_seeds = [int(seed)]
            _record_param_provenance(
                parameters,
                "selection",
                "validation_seeds",
                method="computed_from_seed",
                compute_args={"seed": int(seed)},
            )
    else:
        validation_seeds = [int(s) for s in validation_seeds]
        parameters.setdefault("selection", {})["validation_seeds"] = validation_seeds
        _record_param_provenance(
            parameters,
            "selection",
            "validation_seeds",
            method="explicit_cli",
            compute_args={"source": "cli"},
        )

    if validation_min_distances is None:
        validation_min_distances = [float(config_min_distance_km)]
    else:
        validation_min_distances = [float(d) for d in validation_min_distances]

    resolved_sampler, sampler_source, sampler_artifact_path = _resolve_optuna_sampler(
        config=parameters,
        output_dir=output_dir,
        n_trials=n_trials,
        n_samples=resolved_n_samples,
        validation_seeds=validation_seeds,
        compute_params=compute_params,
        dry_run=dry_run,
    )
    if resolved_sampler is not None:
        parameters.setdefault("selection", {})["optuna_sampler"] = resolved_sampler
    elif not skip_optimization and not dry_run:
        raise ValueError(
            "Optuna sampler unresolved. Set selection.optuna_sampler policy, "
            "provide selected_sampler.json artifact, or run with --compute-params."
        )
    _record_param_provenance(
        parameters,
        "selection",
        "optuna_sampler",
        method=sampler_source,
        source_file=sampler_artifact_path,
        source_hash=compute_file_sha256(Path(sampler_artifact_path))
        if sampler_artifact_path and Path(sampler_artifact_path).exists()
        else None,
    )

    resolved_exploration_sampler, exploration_sampler_source = _resolve_exploration_sampler(
        config=parameters,
        resolved_optuna_sampler=resolved_sampler,
    )
    if resolved_exploration_sampler is None and not skip_exploration and not dry_run:
        raise ValueError(
            "Exploration sampler unresolved. Set selection.exploration_sampler policy "
            "or provide a resolvable optuna sampler policy/artifact."
        )
    if resolved_exploration_sampler is not None:
        parameters.setdefault("selection", {})[
            "exploration_sampler"
        ] = resolved_exploration_sampler
        _record_param_provenance(
            parameters,
            "selection",
            "exploration_sampler",
            method=exploration_sampler_source,
            compute_args={"resolved_optuna_sampler": resolved_sampler},
        )

    parameters.setdefault("thesis_runtime", {})["n_trials"] = int(n_trials)
    _record_param_provenance(
        parameters,
        "thesis_runtime",
        "n_trials",
        method="explicit_cli",
        compute_args={"source": "cli_or_default"},
        notes="Optuna trial budget for thesis pipeline",
    )

    pre_names_list = list(pre_names) if pre_names is not None else []
    if hamburg:
        pre_names_list.append("Hamburg")
    # Keep deterministic ordering while removing duplicates.
    pre_names_list = list(dict.fromkeys(pre_names_list))
    pre_indices_list = (
        list(dict.fromkeys(int(idx) for idx in pre_indices))
        if pre_indices is not None
        else []
    )

    # Compute adaptive n_lhs if not provided
    n_lhs_source = "explicit_cli" if n_lhs is not None else "computed_adaptive"
    if n_lhs is None:
        try:
            import numpy as np
            import pandas as pd

            if metadata_path.exists():
                n_tiles = len(pd.read_csv(metadata_path))
                n_lhs = max(50, int(2 * np.sqrt(n_tiles)))
                print(f"📊 Adaptive n_lhs computed from dataset: {n_lhs}")
                n_lhs_source = "computed_adaptive_from_metadata"
            else:
                n_lhs = 50
                print(f"⚠️ Metadata not found; using fallback n_lhs={n_lhs}")
                n_lhs_source = "workflow_default_policy"
        except Exception:
            n_lhs = 50
            print(f"⚠️ Could not compute adaptive n_lhs; using fallback n_lhs={n_lhs}")
            n_lhs_source = "workflow_default_policy"

    parameters.setdefault("thesis_runtime", {})["n_lhs"] = int(n_lhs)
    _record_param_provenance(
        parameters,
        "thesis_runtime",
        "n_lhs",
        method=n_lhs_source,
        source_file=str(metadata_path) if n_lhs_source == "computed_adaptive_from_metadata" else None,
        source_hash=compute_file_sha256(metadata_path)
        if n_lhs_source == "computed_adaptive_from_metadata" and metadata_path.exists()
        else None,
        compute_args={"seed": int(seed)},
    )

    # Persist a resolved snapshot before execution for centralized traceability.
    resolved_snapshot_path: Path | None = None
    source_files: dict[str, Any] = {}
    if config_path.exists():
        source_files["active_config"] = {
            "path": str(config_path),
            "sha256": compute_file_sha256(config_path),
        }
    if snapshot_input_path is not None and snapshot_input_path.exists():
        source_files["input_snapshot"] = {
            "path": str(snapshot_input_path),
            "sha256": compute_file_sha256(snapshot_input_path),
        }
    if sampler_artifact_path and Path(sampler_artifact_path).exists():
        source_files["sampler_artifact"] = {
            "path": sampler_artifact_path,
            "sha256": compute_file_sha256(Path(sampler_artifact_path)),
        }
    if (
        computed_selection_source_file
        and Path(computed_selection_source_file).exists()
    ):
        source_files["selection_compute_artifact"] = {
            "path": computed_selection_source_file,
            "sha256": compute_file_sha256(Path(computed_selection_source_file)),
        }

    snapshot = build_snapshot(
        parameters=parameters,
        provenance={"source_files": source_files},
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "parameter_source": parameter_source,
        },
        notes="thesis_pipeline resolved snapshot",
    )
    resolved_snapshot_path = output_dir / f"final_config_{run_timestamp}.yaml"
    write_snapshot(snapshot, resolved_snapshot_path)
    print(f"🧾 Wrote resolved snapshot: {resolved_snapshot_path}")
    stable_snapshot_path: Path | None = None
    if snapshot_config:
        stable_snapshot_path = output_dir / "final_config.yaml"
        stable_snapshot_path.write_text(
            resolved_snapshot_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"🧾 Wrote stable snapshot alias: {stable_snapshot_path}")

    if no_auto_continue:
        print("⏹️  Resolution finished; stopping due to --no-auto-continue.")
        resolved_snapshot_sha256 = (
            compute_file_sha256(resolved_snapshot_path)
            if resolved_snapshot_path and resolved_snapshot_path.exists()
            else None
        )
        try:
            write_run_metadata(
                output_dir=output_dir,
                execution_profile=execution_profile,
                seed=seed,
                command=sys.argv,
                config_path=config_path,
                runtime_state=runtime_state,
                extra={
                    "resolution_only": True,
                    "parameter_source": parameter_source,
                    "resolved_sampler": resolved_sampler,
                    "resolved_sampler_source": sampler_source,
                    "resolved_exploration_sampler": resolved_exploration_sampler,
                    "resolved_exploration_sampler_source": exploration_sampler_source,
                    "resolved_snapshot_path": str(resolved_snapshot_path)
                    if resolved_snapshot_path
                    else None,
                    "stable_snapshot_path": str(stable_snapshot_path)
                    if stable_snapshot_path
                    else None,
                    "resolved_snapshot_sha256": resolved_snapshot_sha256,
                    "snapshot_validation_errors": snapshot_errors,
                    "snapshot_forced": snapshot_forced,
                    "resolved_n_clusters": resolved_n_clusters,
                    "resolved_batch_size": resolved_batch_size,
                    "resolved_umap_components": resolved_umap_components,
                    "resolved_umap_n_neighbors": resolved_umap_n_neighbors,
                    "resolved_umap_random_state": resolved_umap_random_state,
                    "resolved_umap_n_jobs": resolved_umap_n_jobs,
                    "computed_selection_method": computed_selection_method,
                    "computed_selection_source_file": computed_selection_source_file,
                    "computed_selection_source_hash": computed_selection_source_hash,
                },
            )
        except Exception as exc:
            print(f"⚠️ Could not write run metadata: {exc}")
        return True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("🚀 THESIS OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Start: {timestamp}")
    print(f"Output Directory: {output_dir}")
    print(
        f"n_lhs: {n_lhs}, n_samples: {resolved_n_samples} ({n_samples_source}), "
        f"n_trials: {n_trials}"
    )
    print(
        "validation: seeds={}, min_distances={}".format(
            validation_seeds,
            validation_min_distances,
        )
    )
    print(
        "resolved sampler: {} ({})".format(
            resolved_sampler if resolved_sampler is not None else "<workflow-default>",
            sampler_source,
        )
    )
    print(
        "exploration sampler: {} ({})".format(
            resolved_exploration_sampler
            if resolved_exploration_sampler is not None
            else "<unresolved>",
            exploration_sampler_source,
        )
    )
    print(
        "Preselection: names={}, indices={}".format(
            pre_names_list if pre_names_list else None,
            pre_indices_list if pre_indices_list else None,
        )
    )
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
                    selection_n_samples=resolved_n_samples,
                    sampler=resolved_exploration_sampler,
                    seed=seed,
                    metadata_path=metadata_path,
                    min_distance_km=config_min_distance_km,
                    n_clusters=resolved_n_clusters,
                    batch_size=resolved_batch_size,
                    pre_names=pre_names_list if pre_names_list else None,
                    pre_indices=pre_indices_list if pre_indices_list else None,
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
                    n_samples=resolved_n_samples,
                    sampler_name=resolved_sampler,
                    metadata_path=metadata_path,
                    seed=seed,
                    pre_selected_names=pre_names_list if pre_names_list else None,
                    pre_selected_indices=pre_indices_list if pre_indices_list else None,
                    out_dir=output_dir / "optuna",
                    feature_cache_dir=Path("outputs"),
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
                    min_distances=validation_min_distances,
                    seeds=validation_seeds,
                    output_dir=output_dir / "validation",
                    feature_cache_dir=Path("outputs"),
                    n_samples=resolved_n_samples,
                    n_clusters=resolved_n_clusters,
                    batch_size=resolved_batch_size,
                    umap_n_components=resolved_umap_components,
                    umap_n_neighbors=resolved_umap_n_neighbors,
                    umap_random_state=resolved_umap_random_state,
                    umap_n_jobs=resolved_umap_n_jobs,
                    pre_selected_names=pre_names_list if pre_names_list else None,
                    pre_selected_indices=pre_indices_list if pre_indices_list else None,
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
        resolved_snapshot_sha256 = (
            compute_file_sha256(resolved_snapshot_path)
            if resolved_snapshot_path and resolved_snapshot_path.exists()
            else None
        )
        write_run_metadata(
            output_dir=output_dir,
            execution_profile=execution_profile,
            seed=seed,
            command=sys.argv,
            config_path=Path("config/pipeline_config.yaml"),
            runtime_state=runtime_state,
            extra={
                "n_lhs": n_lhs,
                "n_samples": resolved_n_samples,
                "n_samples_source": n_samples_source,
                "parameter_source": parameter_source,
                "compute_params": bool(compute_params),
                "resolved_sampler": resolved_sampler,
                "resolved_sampler_source": sampler_source,
                "sampler_artifact_path": sampler_artifact_path,
                "resolved_exploration_sampler": resolved_exploration_sampler,
                "resolved_exploration_sampler_source": exploration_sampler_source,
                "resolved_snapshot_path": str(resolved_snapshot_path)
                if resolved_snapshot_path
                else None,
                "stable_snapshot_path": str(stable_snapshot_path)
                if stable_snapshot_path
                else None,
                "resolved_snapshot_sha256": resolved_snapshot_sha256,
                "snapshot_validation_errors": snapshot_errors,
                "snapshot_forced": snapshot_forced,
                "n_trials": n_trials,
                "resolved_n_clusters": resolved_n_clusters,
                "resolved_batch_size": resolved_batch_size,
                "resolved_umap_components": resolved_umap_components,
                "resolved_umap_n_neighbors": resolved_umap_n_neighbors,
                "resolved_umap_random_state": resolved_umap_random_state,
                "resolved_umap_n_jobs": resolved_umap_n_jobs,
                "computed_selection_method": computed_selection_method,
                "computed_selection_source_file": computed_selection_source_file,
                "computed_selection_source_hash": computed_selection_source_hash,
                "skip_exploration": skip_exploration,
                "skip_optimization": skip_optimization,
                "skip_validation": skip_validation,
                "dry_run": dry_run,
                "validation_seeds": validation_seeds,
                "validation_min_distances": validation_min_distances,
                "pre_selected_names": pre_names_list if pre_names_list else None,
                "pre_selected_indices": pre_indices_list if pre_indices_list else None,
                "hamburg_shortcut": bool(hamburg),
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
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Target selected tiles (explicit > config > autoscale artifact)",
        },
        "n_trials": {
            "type": int,
            "default": 100,
            "help": "Number of Optuna trials",
        },
        "compute_params": {
            "type": bool,
            "action": "store_true",
            "help": "Compute unresolved parameters and record provenance",
        },
        "use_params": {
            "type": str,
            "default": None,
            "help": "Load parameter snapshot YAML and validate before run",
        },
        "snapshot_config": {
            "type": bool,
            "action": "store_true",
            "help": "Write resolved final_config snapshot before run",
        },
        "no_auto_continue": {
            "type": bool,
            "action": "store_true",
            "help": "Stop after parameter resolution/snapshot stage",
        },
        "force": {
            "type": bool,
            "action": "store_true",
            "help": "Continue despite snapshot/provenance validation mismatch",
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
            "help": "Output directory (default: outputs/runs/thesis_pipeline_<timestamp>)",
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
            "help": "Pre-selected tile indices",
        },
        "hamburg": {
            "type": bool,
            "action": "store_true",
            "help": "Add Hamburg to pre-names",
        },
        "validation_seeds": {
            "type": int,
            "nargs": "+",
            "default": None,
            "help": "Validation seed list (quick gate default: seed only)",
        },
        "validation_min_distances": {
            "type": float,
            "nargs": "+",
            "default": None,
            "help": "Validation min_distance list in km (quick gate default: policy value)",
        },
    },
)
def main(
    n_lhs: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_trials: int = 100,
    compute_params: bool = False,
    use_params: Optional[str] = None,
    snapshot_config: bool = False,
    no_auto_continue: bool = False,
    force: bool = False,
    skip_exploration: bool = False,
    skip_optimization: bool = False,
    skip_validation: bool = False,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    execution_profile: str = "default",
    seed: int = 42,
    pre_names: Optional[list[str]] = None,
    pre_indices: Optional[list[int]] = None,
    hamburg: bool = False,
    validation_seeds: Optional[list[int]] = None,
    validation_min_distances: Optional[list[float]] = None,
) -> int:
    """CLI entry point for thesis pipeline."""
    # Convert str path to Path object
    output_dir_path = Path(output_dir) if output_dir else None

    # Run the pipeline
    success = run_thesis_pipeline(
        n_lhs=n_lhs,
        n_samples=n_samples,
        n_trials=n_trials,
        compute_params=compute_params,
        use_params=Path(use_params) if use_params else None,
        snapshot_config=snapshot_config,
        no_auto_continue=no_auto_continue,
        force=force,
        skip_exploration=skip_exploration,
        skip_optimization=skip_optimization,
        skip_validation=skip_validation,
        dry_run=dry_run,
        output_dir=output_dir_path,
        execution_profile=execution_profile,
        seed=seed,
        pre_names=pre_names,
        pre_indices=pre_indices,
        hamburg=hamburg,
        validation_seeds=validation_seeds,
        validation_min_distances=validation_min_distances,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
