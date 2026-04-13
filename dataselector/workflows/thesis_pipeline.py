"""Master pipeline for thesis optimization."""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from dataselector.cli_decorators import cli_command
from dataselector.runtime import (
    activate_repro_mode,
    report_exception,
    write_run_metadata,
)
from dataselector.runtime.parameter_snapshot import (
    build_snapshot,
    compute_file_sha256,
    load_snapshot,
    validate_snapshot_file,
    write_snapshot,
)
from dataselector.workflows._selection_target import resolve_selection_n_samples
from dataselector.workflows.annotation_plan import run_thesis_build_annotation_plan
from dataselector.workflows.handoff_bundle import (
    prepare_patch_handoff,
    prepare_tile_handoff,
    verify_patch_handoff,
    verify_tile_handoff,
)

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


def _ensure_provenance_section(
    parameters: dict[str, Any], section: str
) -> dict[str, Any]:
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


def _parse_pooling(value: Any) -> str:
    pooled = str(value).strip().lower()
    if pooled not in {"cls", "mean", "global_avg"}:
        raise ValueError(
            f"feature_extraction.pooling must be one of cls|mean|global_avg (got {value!r})"
        )
    return pooled


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


def _materialize_sampler_resolution_artifact(
    *,
    output_dir: Path,
    sampler: str,
    source: str,
    source_artifact: str | None,
) -> Path:
    """Persist a run-local sampler resolution artifact for contract evidence."""
    artifact_dir = output_dir / "parameter_resolution" / "sampler_resolution"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "selected_sampler.json"
    payload: dict[str, Any] = {
        "best": sampler,
        "sampler": sampler,
        "source": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if source_artifact:
        payload["source_artifact"] = source_artifact
        source_path = Path(source_artifact)
        if source_path.exists():
            payload["source_artifact_sha256"] = compute_file_sha256(source_path)

    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return artifact_path


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
        output_dir
        / "parameter_resolution"
        / "sampler_resolution"
        / "selected_sampler.json",
        output_dir / "selected_sampler.json",
        output_dir / "sampler_resolution" / "selected_sampler.json",
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

        compare_out = output_dir / "parameter_resolution" / "sampler_resolution"
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


def _parse_case_attach_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    allowed = {"append_unique", "append_all"}
    if mode not in allowed:
        raise ValueError(
            "selection.case_attach_mode must be one of append_unique|append_all "
            f"(got {value!r})"
        )
    return mode


def _parse_selection_authority(value: Any) -> str:
    mode = str(value).strip().lower()
    allowed = {"snapshot_primary", "materialized_csv_primary"}
    if mode not in allowed:
        raise ValueError(
            "selection.selection_authority must be one of "
            "snapshot_primary|materialized_csv_primary "
            f"(got {value!r})"
        )
    return mode


def _parse_objective_authority(value: Any) -> str:
    mode = str(value).strip().lower()
    allowed = {"unified_normalized", "legacy_lexicographic"}
    if mode not in allowed:
        raise ValueError(
            "selection.objective_authority must be one of "
            "unified_normalized|legacy_lexicographic "
            f"(got {value!r})"
        )
    return mode


def _dedupe_str_list(values: list[Any] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _resolve_named_indices(
    metadata: pd.DataFrame,
    names: list[str],
    *,
    alias_map: dict[str, list[str]] | None = None,
) -> tuple[list[int], dict[int, str], list[str]]:
    resolved: list[int] = []
    index_to_name: dict[int, str] = {}
    unresolved: list[str] = []
    aliases = alias_map or {"hamburg": ["KDR_146"]}

    for nm in names:
        text = str(nm).strip()
        if not text:
            continue
        terms = [text]
        terms.extend(aliases.get(text.lower(), []))
        mask = pd.Series(False, index=metadata.index)
        for term in terms:
            needle = str(term).strip().lower()
            if not needle:
                continue
            if "longName" in metadata.columns:
                mask = mask | metadata["longName"].astype(str).str.lower().str.contains(
                    needle
                )
            if "shortName" in metadata.columns:
                mask = mask | (metadata["shortName"].astype(str).str.lower() == needle)
            if "city" in metadata.columns:
                mask = mask | (metadata["city"].astype(str).str.lower() == needle)
        idxs = [int(i) for i in mask[mask].index.tolist()]
        if not idxs:
            unresolved.append(text)
            continue
        for idx in idxs:
            if idx not in index_to_name:
                index_to_name[idx] = text
            resolved.append(idx)
    resolved = list(dict.fromkeys(int(i) for i in resolved))
    return resolved, index_to_name, unresolved


def _find_primary_selection_csv(output_dir: Path) -> tuple[Path | None, str]:
    tuning_meta_json = output_dir / "tuning_weights" / "meta.json"
    tuning_weights_dir = output_dir / "tuning_weights"
    selected_tiles_file: Path | None = None
    selected_from = "not_available"

    if tuning_meta_json.exists():
        try:
            tuning_meta = json.loads(tuning_meta_json.read_text(encoding="utf-8"))
            best_metrics = tuning_meta.get("best_metrics", {})
            alpha = best_metrics.get("alpha")
            beta = best_metrics.get("beta")
            gamma = best_metrics.get("gamma")
            if alpha is not None and beta is not None and gamma is not None:
                exact_name = f"selection_a{alpha}_b{beta}_g{gamma}.csv"
                exact_path = tuning_weights_dir / exact_name
                if exact_path.exists():
                    selected_tiles_file = exact_path
                else:
                    pattern = f"selection_a{float(alpha):.6f}*_b{float(beta):.6f}*_g{float(gamma):.6f}*.csv"
                    candidates = sorted(tuning_weights_dir.glob(pattern))
                    if candidates:
                        selected_tiles_file = candidates[0]
                if selected_tiles_file is not None:
                    selected_from = "tuning_weights_best_metrics"
        except Exception:
            selected_tiles_file = None

    if selected_tiles_file is None:
        validation_selection_files = sorted(
            (output_dir / "validation").glob("selection_a*_b*_g*_d*_s*.csv")
        )
        if validation_selection_files:
            selected_tiles_file = validation_selection_files[0]
            selected_from = "validation_fallback"

    return selected_tiles_file, selected_from


def _parse_selection_weights_from_filename(
    path_value: str | None,
) -> dict[str, float] | None:
    if not path_value:
        return None
    match = re.search(
        r"selection_a(?P<a>[-+0-9.eE]+?)_b(?P<b>[-+0-9.eE]+?)_g(?P<g>[-+0-9.eE]+?)(?:\.csv)?$",
        Path(path_value).name,
    )
    if not match:
        return None
    try:
        return {
            "alpha_visual": float(match.group("a")),
            "beta_spatial": float(match.group("b")),
            "gamma_temporal": float(match.group("g")),
        }
    except Exception:
        return None


def _sanitize_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a detached DataFrame copy safe for concat/export operations."""

    cleaned = df.copy()
    if hasattr(cleaned, "attrs"):
        cleaned.attrs = {}
    return cleaned


def _write_crs_provenance_audit(
    *,
    output_dir: Path,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    from dataselector.data.crs_provenance import audit_crs_provenance

    data_quality_dir = output_dir / "data_quality"
    data_quality_dir.mkdir(parents=True, exist_ok=True)
    audit_path = data_quality_dir / "crs_provenance_audit.csv"
    audit_df, summary = audit_crs_provenance(metadata)
    audit_df.to_csv(audit_path, index=False)
    return {"crs_provenance_audit_path": str(audit_path), **summary}


def _materialize_snapshot_primary_core_selection(
    *,
    output_dir: Path,
    metadata_path: Path,
    config_path: Path,
    resolved_feature_config: dict[str, Any],
    n_samples: int,
    metric: str,
    alpha_visual: float,
    beta_spatial: float,
    gamma_temporal: float,
    min_distance_km: float,
    spatial_constraint: bool,
    use_multi_criteria: bool,
    use_constraint_integration: bool,
    random_state: int,
    pre_selected_names: list[str] | None,
    pre_selected_indices: list[int] | None,
    tile_exclusion_policy: str | Path | None,
    apply_tile_exclusion: bool,
) -> tuple[pd.DataFrame, Path, list[str]]:
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.diversity_selector import DiversitySelector

    metadata = load_metadata(
        metadata_path,
        resolve_images=False,
        tile_exclusion_policy=tile_exclusion_policy,
        apply_tile_exclusion=apply_tile_exclusion,
    )
    feature_cfg = {
        str(k): v
        for k, v in dict(resolved_feature_config or {}).items()
        if not str(k).startswith("_")
    }
    resolved_batch_size = int(feature_cfg.get("batch_size", 16))
    features = load_or_extract_features(
        out_dir=output_dir / "parameter_resolution",
        csv_meta=str(metadata_path),
        batch_size=resolved_batch_size,
        cache=True,
        cache_mode=os.getenv("DATASELECTOR_FEATURE_CACHE_MODE", "read_write"),
        config_path=config_path,
        resolved_feature_config=feature_cfg,
        tile_exclusion_policy=tile_exclusion_policy,
        apply_tile_exclusion=apply_tile_exclusion,
    )
    if len(features) != len(metadata):
        raise ValueError(
            "Feature/metadata length mismatch for snapshot-primary selection: "
            f"features={len(features)} metadata={len(metadata)}"
        )

    raw_pre_indices = (
        list(dict.fromkeys(int(i) for i in (pre_selected_indices or [])))
        if pre_selected_indices
        else []
    )
    pre_name_list = _dedupe_str_list(pre_selected_names or [])
    resolved_name_indices, _, unresolved_pre_names = _resolve_named_indices(
        metadata,
        pre_name_list,
    )
    combined_pre = list(dict.fromkeys(raw_pre_indices + resolved_name_indices))

    selector = DiversitySelector(
        n_samples=int(n_samples),
        metric=str(metric),
        random_state=int(random_state),
        use_constraint_integration=bool(use_constraint_integration),
        use_multi_criteria=bool(use_multi_criteria),
    )
    selected_idx = selector.select(
        features=features,
        metadata=metadata,
        spatial_constraint=bool(spatial_constraint),
        min_distance_km=float(min_distance_km),
        alpha_visual=float(alpha_visual),
        beta_spatial=float(beta_spatial),
        gamma_temporal=float(gamma_temporal),
        pre_selected=combined_pre if combined_pre else None,
    )

    if len(selected_idx) == 0:
        core_df = metadata.iloc[[]].copy().reset_index(drop=True)
    else:
        core_df = metadata.iloc[list(selected_idx)].copy().reset_index(drop=True)
    core_df = _sanitize_export_dataframe(core_df)
    core_df["selection_rank"] = range(len(core_df))
    if len(core_df) > 0:
        core_df["selection_backend"] = str(
            getattr(selector, "selection_backend", "unknown")
        )

    snapshot_core_path = output_dir / "selection_snapshot_primary.csv"
    core_df.to_csv(snapshot_core_path, index=False)
    return core_df, snapshot_core_path, unresolved_pre_names


def _materialize_core_case_artifacts(
    *,
    output_dir: Path,
    metadata_path: Path,
    config_path: Path,
    case_names: list[str],
    case_exclude_from_core: bool,
    case_attach_mode: str,
    selection_authority: str,
    objective_authority: str,
    selection_params: dict[str, Any],
    resolved_feature_config: dict[str, Any],
    pre_selected_names: list[str] | None = None,
    pre_selected_indices: list[int] | None = None,
    tile_exclusion_policy: str | Path | None = None,
    apply_tile_exclusion: bool = False,
) -> dict[str, Any]:
    from dataselector.data.io import load_metadata

    metadata = load_metadata(
        metadata_path,
        resolve_images=False,
        tile_exclusion_policy=tile_exclusion_policy,
        apply_tile_exclusion=apply_tile_exclusion,
    )

    unresolved_pre_names: list[str] = []
    if selection_authority == "snapshot_primary":
        core_raw, primary_sel_path, unresolved_pre_names = (
            _materialize_snapshot_primary_core_selection(
                output_dir=output_dir,
                metadata_path=metadata_path,
                config_path=config_path,
                resolved_feature_config=resolved_feature_config,
                n_samples=int(selection_params["n_samples"]),
                metric=str(selection_params["metric"]),
                alpha_visual=float(selection_params["alpha_visual"]),
                beta_spatial=float(selection_params["beta_spatial"]),
                gamma_temporal=float(selection_params["gamma_temporal"]),
                min_distance_km=float(selection_params["min_distance_km"]),
                spatial_constraint=bool(selection_params["spatial_constraint"]),
                use_multi_criteria=bool(selection_params["use_multi_criteria"]),
                use_constraint_integration=bool(
                    selection_params["use_constraint_integration"]
                ),
                random_state=int(selection_params["random_state"]),
                pre_selected_names=pre_selected_names,
                pre_selected_indices=pre_selected_indices,
                tile_exclusion_policy=tile_exclusion_policy,
                apply_tile_exclusion=apply_tile_exclusion,
            )
        )
        selection_source = "snapshot_primary_selection"
    else:
        # Legacy compatibility mode: freeze dataset claims to recorded selection CSV.
        primary_sel_path, selection_source = _find_primary_selection_csv(output_dir)
        if primary_sel_path is not None and primary_sel_path.exists():
            core_raw = pd.read_csv(primary_sel_path)
        else:
            core_raw = pd.DataFrame(columns=list(metadata.columns))
    core_raw = _sanitize_export_dataframe(core_raw)

    if "selection_rank" in core_raw.columns:
        core_raw["selection_rank"] = (
            pd.to_numeric(core_raw["selection_rank"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        core_raw = core_raw.sort_values("selection_rank").reset_index(drop=True)
    else:
        core_raw = core_raw.reset_index(drop=True)
        core_raw["selection_rank"] = range(len(core_raw))

    case_indices, idx_to_case_name, unresolved_cases = _resolve_named_indices(
        metadata,
        case_names,
    )
    case_df = (
        metadata.iloc[case_indices].copy() if case_indices else metadata.iloc[[]].copy()
    )
    case_df = _sanitize_export_dataframe(case_df)
    if len(case_df) > 0:
        case_df["case_name"] = [idx_to_case_name.get(int(i), "") for i in case_df.index]
        case_df["case_reason"] = "manual_case_tile"
        case_df = case_df.reset_index(drop=True)
    case_df["selection_rank"] = range(len(case_df))

    def _key_series(df: pd.DataFrame) -> pd.Series:
        if "shortName" in df.columns:
            return df["shortName"].astype(str).str.lower()
        if "longName" in df.columns:
            return df["longName"].astype(str).str.lower()
        return pd.Series([str(i) for i in range(len(df))], index=df.index)

    core_keys = _key_series(core_raw) if len(core_raw) > 0 else pd.Series(dtype=str)
    case_keys = set(_key_series(case_df).tolist()) if len(case_df) > 0 else set()

    if case_exclude_from_core and len(core_raw) > 0 and case_keys:
        keep_mask = ~core_keys.isin(case_keys)
        core_df = core_raw.loc[keep_mask].reset_index(drop=True)
    else:
        core_df = core_raw.copy().reset_index(drop=True)
    core_df = _sanitize_export_dataframe(core_df)
    core_df["selection_rank"] = range(len(core_df))

    if case_attach_mode == "append_all":
        append_df = case_df.copy()
    else:
        if len(case_df) == 0:
            append_df = case_df.copy()
        else:
            existing_keys = (
                set(_key_series(core_df).tolist()) if len(core_df) > 0 else set()
            )
            append_mask = ~_key_series(case_df).isin(existing_keys)
            append_df = case_df.loc[append_mask].reset_index(drop=True)
    append_df = _sanitize_export_dataframe(append_df)

    if len(core_df) == 0:
        final_df = append_df.copy().reset_index(drop=True)
    elif len(append_df) == 0:
        final_df = core_df.copy().reset_index(drop=True)
    else:
        final_df = pd.concat([core_df, append_df], ignore_index=True)
    final_df = _sanitize_export_dataframe(final_df)
    if len(final_df) > 0:
        final_df["selection_rank"] = range(len(final_df))

    core_path = output_dir / "selection_core.csv"
    case_path = output_dir / "selection_case.csv"
    final_path = output_dir / "selection_final_with_cases.csv"
    contract_path = output_dir / "selection_contract.json"

    core_df.to_csv(core_path, index=False)
    case_df.to_csv(case_path, index=False)
    final_df.to_csv(final_path, index=False)

    contract = {
        "contract_version": "core_case_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_authority": selection_authority,
        "objective_authority": objective_authority,
        "selection_source": selection_source,
        "selection_source_file": None,
        "selection_weights": (
            {
                "alpha_visual": float(selection_params["alpha_visual"]),
                "beta_spatial": float(selection_params["beta_spatial"]),
                "gamma_temporal": float(selection_params["gamma_temporal"]),
            }
            if selection_authority == "snapshot_primary"
            else (
                _parse_selection_weights_from_filename(str(primary_sel_path))
                if primary_sel_path is not None
                else None
            )
        ),
        "case_tile_names": case_names,
        "case_exclude_from_core": bool(case_exclude_from_core),
        "case_attach_mode": case_attach_mode,
        "core_count_raw": int(len(core_raw)),
        "core_count": int(len(core_df)),
        "case_count_resolved": int(len(case_df)),
        "case_count_attached": int(len(append_df)),
        "case_count": int(len(case_df)),
        "final_count": int(len(final_df)),
        "unresolved_case_names": unresolved_cases,
        "unresolved_pre_selected_names": unresolved_pre_names,
    }
    if primary_sel_path is not None and primary_sel_path.exists():
        try:
            contract["selection_source_file"] = str(
                primary_sel_path.relative_to(output_dir)
            )
        except Exception:
            contract["selection_source_file"] = str(primary_sel_path)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return {
        "selection_core_path": str(core_path),
        "selection_case_path": str(case_path),
        "selection_final_with_cases_path": str(final_path),
        "selection_contract_path": str(contract_path),
        **contract,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_handoff_root(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _freeze_boundary_hashes(
    *,
    output_dir: Path,
    resolved_snapshot_path: Path | None,
) -> dict[str, str]:
    protected_paths = {
        "selection_core.csv": output_dir / "selection_core.csv",
        "selection_case.csv": output_dir / "selection_case.csv",
        "selection_final_with_cases.csv": output_dir / "selection_final_with_cases.csv",
        "selection_contract.json": output_dir / "selection_contract.json",
    }
    if resolved_snapshot_path is not None:
        protected_paths["resolved_snapshot"] = resolved_snapshot_path

    hashes: dict[str, str] = {}
    missing: list[str] = []
    for label, path in protected_paths.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
            continue
        hashes[label] = compute_file_sha256(path)
    if missing:
        raise FileNotFoundError(
            "Phase 5 freeze-boundary inputs missing: " + "; ".join(missing)
        )
    return hashes


def _run_phase5_annotation_handoffs(
    *,
    output_dir: Path,
    resolved_snapshot_path: Path | None,
    handoff_root: str | Path,
    patches_per_tile: int,
    patch_include_case: bool,
    patch_id_file: str | Path | None,
    tile_exclusion_policy: Path | None,
) -> dict[str, Any]:
    handoff_root_path = _resolve_handoff_root(handoff_root)
    run_id = output_dir.name
    patch_suffix = "final" if patch_include_case else "core"
    tile_handoff_dir = handoff_root_path / run_id
    patch_handoff_dir = handoff_root_path / f"{run_id}_patches_{patch_suffix}"

    pre_hashes = _freeze_boundary_hashes(
        output_dir=output_dir,
        resolved_snapshot_path=resolved_snapshot_path,
    )

    tile_prepare = prepare_tile_handoff(
        run_dir=output_dir,
        out_dir=tile_handoff_dir,
        repo_root=_repo_root(),
        tile_exclusion_policy=tile_exclusion_policy
        or "config/tile_exclusion_policy.yaml",
    )
    tile_verify = verify_tile_handoff(
        handoff_dir=tile_handoff_dir,
        repo_root=_repo_root(),
        tile_exclusion_policy=tile_exclusion_policy
        or "config/tile_exclusion_policy.yaml",
    )

    annotation_summary = run_thesis_build_annotation_plan(
        run_dir=output_dir,
        patches_per_tile=int(patches_per_tile),
        include_case=bool(patch_include_case),
    )
    patch_prepare = prepare_patch_handoff(
        run_dir=output_dir,
        out_dir=patch_handoff_dir,
        patch_id_file=patch_id_file,
        repo_root=_repo_root(),
        tile_exclusion_policy=tile_exclusion_policy
        or "config/tile_exclusion_policy.yaml",
    )
    patch_verify = verify_patch_handoff(
        handoff_dir=patch_handoff_dir,
        repo_root=_repo_root(),
        tile_exclusion_policy=tile_exclusion_policy
        or "config/tile_exclusion_policy.yaml",
    )

    post_hashes = _freeze_boundary_hashes(
        output_dir=output_dir,
        resolved_snapshot_path=resolved_snapshot_path,
    )
    if pre_hashes != post_hashes:
        raise RuntimeError(
            "Phase 5 freeze-boundary integrity violation: protected scientific artifacts changed."
        )

    annotation_contract_path = (
        Path(annotation_summary["annotation_dataset_contract_json"])
        if annotation_summary.get("annotation_dataset_contract_json")
        else output_dir / "annotation_plan" / "annotation_dataset_contract.json"
    )
    return {
        "build_handoffs": True,
        "handoff_root": str(handoff_root_path),
        "tile_handoff_dir": str(tile_handoff_dir),
        "tile_handoff_manifest_path": str(tile_handoff_dir / "handoff_manifest.json"),
        "tile_handoff_selection_count": tile_prepare.get("selection_count"),
        "tile_handoff_verify": tile_verify,
        "annotation_plan_dir": str(Path(annotation_summary["output_dir"])),
        "annotation_dataset_contract_path": str(annotation_contract_path),
        "patches_per_tile": int(patches_per_tile),
        "patch_include_case": bool(patch_include_case),
        "patch_selection_group": "final" if patch_include_case else "core",
        "patches_total": annotation_summary.get("patches_total"),
        "patches_qc_passed": annotation_summary.get("patches_qc_passed"),
        "patches_qc_rejected": annotation_summary.get("patches_qc_rejected"),
        "patch_handoff_dir": str(patch_handoff_dir),
        "patch_handoff_manifest_path": str(
            patch_handoff_dir / "patch_handoff_manifest.json"
        ),
        "patch_handoff_selection_count": patch_prepare.get("selection_count"),
        "patch_id_filter_path": patch_prepare.get("patch_id_filter_path", ""),
        "patch_handoff_verify": patch_verify,
        "phase5_freeze_boundary_verified": True,
        "phase5_freeze_boundary_hashes": pre_hashes,
        "phase5_freeze_boundary_hashes_post": post_hashes,
    }


def run_thesis_pipeline(
    n_lhs: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_trials: int = 370,
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
    config_path: Optional[Path] = None,
    cache_mode: str = "read_write",
    execution_profile: str = "default",
    seed: int = 42,
    strict_scientific: bool = True,
    pre_names: Optional[list[str]] = None,
    pre_indices: Optional[list[int]] = None,
    hamburg: bool = False,
    case_names: Optional[list[str]] = None,
    case_exclude_from_core: Optional[bool] = None,
    case_attach_mode: Optional[str] = None,
    validation_seeds: Optional[list[int]] = None,
    validation_min_distances: Optional[list[float]] = None,
    validation_replicate_mode: Optional[str] = None,
    validation_n_bootstrap: Optional[int] = None,
    validation_bootstrap_sample_frac: Optional[float] = None,
    tile_exclusion_policy: Optional[Path] = None,
    apply_tile_exclusion: Optional[bool] = None,
    build_handoffs: bool = False,
    patches_per_tile: int = 2,
    patch_include_case: bool = False,
    patch_id_file: str | Path | None = None,
    handoff_root: str = "handoff",
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
        case_names: Optional case tile names (appended after core selection)
        case_exclude_from_core: Exclude case tiles from core selection
        case_attach_mode: How to attach case tiles (`append_unique` or `append_all`)
        validation_seeds: Optional validation seed list (quick gate default: [seed])
        validation_min_distances: Optional min_distance list for validation
        validation_replicate_mode: Validation replicate mode (`seed_replay` or `bootstrap_candidates`)
        validation_n_bootstrap: Number of bootstrap replicates for validation
        validation_bootstrap_sample_frac: Candidate sample fraction for bootstrap replicates
        tile_exclusion_policy: Explicit tile policy path for candidate-pool filtering/flagging
        apply_tile_exclusion: Explicit toggle for candidate-pool exclusion policy
        build_handoffs: Enable optional post-freeze Phase 5 handoff bundle
        patches_per_tile: Annotation-plan patches per selected tile for Phase 5
        patch_include_case: Include case tiles in the Phase 5 patch plan/handoff
        patch_id_file: Optional plain-text patch-id allowlist for a filtered Phase-5 patch handoff
        handoff_root: Root directory for tile/patch handoff bundles

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
    feature_cache_dir = output_dir / "parameter_resolution"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path("data/new_all_tiles.csv")
    config_path = (
        Path(config_path)
        if config_path is not None
        else Path("config/pipeline_config.yaml")
    )
    os.environ["DATASELECTOR_ACTIVE_CONFIG"] = str(config_path)
    os.environ["DATASELECTOR_FEATURE_CACHE_MODE"] = str(cache_mode)
    tile_policy_path = (
        Path(tile_exclusion_policy)
        if tile_exclusion_policy is not None
        else (
            Path(os.environ["DATASELECTOR_TILE_EXCLUSION_POLICY"])
            if os.environ.get("DATASELECTOR_TILE_EXCLUSION_POLICY")
            else None
        )
    )
    apply_tile_exclusion_flag = (
        _parse_bool(apply_tile_exclusion, label="apply_tile_exclusion")
        if apply_tile_exclusion is not None
        else os.environ.get("DATASELECTOR_APPLY_TILE_EXCLUSION", "0") == "1"
    )
    build_handoffs_flag = bool(build_handoffs)
    patches_per_tile_resolved = _parse_positive_int(
        patches_per_tile, label="patches_per_tile"
    )
    patch_include_case_flag = _parse_bool(
        patch_include_case, label="patch_include_case"
    )
    patch_id_file_resolved = (
        str(
            (
                Path(str(patch_id_file))
                if Path(str(patch_id_file)).is_absolute()
                else (_repo_root() / Path(str(patch_id_file)))
            ).resolve()
        )
        if patch_id_file is not None and str(patch_id_file).strip()
        else ""
    )
    handoff_root_resolved = str(_resolve_handoff_root(handoff_root))
    if tile_policy_path is not None:
        os.environ["DATASELECTOR_TILE_EXCLUSION_POLICY"] = str(tile_policy_path)
    os.environ["DATASELECTOR_APPLY_TILE_EXCLUSION"] = (
        "1" if apply_tile_exclusion_flag else "0"
    )
    if execution_profile == "thesis_repro":
        os.environ["DATASELECTOR_STRICT_CRS"] = "1"
        os.environ["DATASELECTOR_STRICT_EXPLICIT_CRS"] = "1"
        os.environ["DATASELECTOR_ALLOW_HEURISTIC_CRS_FALLBACK"] = "0"
        os.environ["DATASELECTOR_METRIC_EPSG"] = "25832"
    else:
        os.environ["DATASELECTOR_STRICT_CRS"] = "0"
        os.environ["DATASELECTOR_STRICT_EXPLICIT_CRS"] = "0"
        os.environ["DATASELECTOR_ALLOW_HEURISTIC_CRS_FALLBACK"] = "1"
        os.environ["DATASELECTOR_METRIC_EPSG"] = "25832"

    metadata_crs_info: dict[str, Any] = {}
    metadata_crs_audit_info: dict[str, Any] = {"crs_provenance_audit_path": None}
    metadata_tile_exclusion_info: dict[str, Any] = {
        "tile_exclusions_applied": None,
        "tile_exclusions_count": None,
        "tile_excluded_shortnames": None,
        "tile_flagged_count": None,
        "tile_flagged_shortnames": None,
        "tile_flagged_classes": None,
        "tile_flagged_caveats": None,
        "tile_exclusion_policy_sha256": None,
        "effective_tile_count": None,
    }
    exception_records: list[dict[str, Any]] = []
    exceptions_log_path: str | None = None
    try:
        from dataselector.data.io import load_metadata

        md_audit = load_metadata(
            metadata_path,
            resolve_images=False,
            strict_metric_crs=False,
            strict_explicit_crs=False,
            allow_heuristic_crs_fallback=True,
            tile_exclusion_policy=tile_policy_path,
            apply_tile_exclusion=apply_tile_exclusion_flag,
        )
        metadata_crs_audit_info = _write_crs_provenance_audit(
            output_dir=output_dir,
            metadata=md_audit,
        )
        if execution_profile == "thesis_repro" and not bool(
            metadata_crs_audit_info.get("crs_strict_ready", False)
        ):
            raise RuntimeError(
                "Explicit CRS audit failed for thesis_repro: "
                f"status={metadata_crs_audit_info.get('crs_provenance_status')}, "
                f"missing={metadata_crs_audit_info.get('crs_missing_explicit_count')}, "
                f"heuristic={metadata_crs_audit_info.get('crs_heuristic_fallback_count')}, "
                f"mismatches={metadata_crs_audit_info.get('crs_consistency_issue_count')}"
            )
        md_preview = (
            load_metadata(
                metadata_path,
                resolve_images=False,
                strict_metric_crs=True,
                strict_explicit_crs=True,
                allow_heuristic_crs_fallback=False,
                tile_exclusion_policy=tile_policy_path,
                apply_tile_exclusion=apply_tile_exclusion_flag,
            )
            if execution_profile == "thesis_repro"
            else md_audit
        )
        metadata_crs_info = {
            "source_crs_declared": (
                md_preview.attrs.get("source_crs_declared")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "source_crs": (
                md_preview.attrs.get("source_crs")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "metric_crs": (
                md_preview.attrs.get("metric_crs")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "crs_source": (
                md_preview.attrs.get("crs_source")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "crs_provenance": (
                md_preview.attrs.get("crs_provenance")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "crs_explicit": (
                bool(md_preview.attrs.get("crs_explicit"))
                if hasattr(md_preview, "attrs")
                else False
            ),
            "transform_applied": (
                bool(md_preview.attrs.get("transform_applied"))
                if hasattr(md_preview, "attrs")
                else False
            ),
            "crs_provenance_status": (
                md_preview.attrs.get("crs_provenance_status")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "crs_explicit_tile_count": (
                int(md_preview.attrs.get("crs_explicit_tile_count", 0))
                if hasattr(md_preview, "attrs")
                else 0
            ),
            "crs_heuristic_fallback_count": (
                int(md_preview.attrs.get("crs_heuristic_fallback_count", 0))
                if hasattr(md_preview, "attrs")
                else 0
            ),
            "crs_missing_explicit_count": (
                int(md_preview.attrs.get("crs_missing_explicit_count", 0))
                if hasattr(md_preview, "attrs")
                else 0
            ),
            "crs_consistency_issue_count": (
                int(md_preview.attrs.get("crs_consistency_issue_count", 0))
                if hasattr(md_preview, "attrs")
                else 0
            ),
            "crs_consistency_issue_shortnames": (
                list(md_preview.attrs.get("crs_consistency_issue_shortnames", []))
                if hasattr(md_preview, "attrs")
                else []
            ),
            "crs_unique_explicit_source_crs": (
                list(md_preview.attrs.get("crs_unique_explicit_source_crs", []))
                if hasattr(md_preview, "attrs")
                else []
            ),
            "crs_explicit_source_crs": (
                md_preview.attrs.get("crs_explicit_source_crs")
                if hasattr(md_preview, "attrs")
                else None
            ),
            "crs_strict_ready": (
                bool(md_preview.attrs.get("crs_strict_ready"))
                if hasattr(md_preview, "attrs")
                else False
            ),
        }
        attrs = md_preview.attrs if hasattr(md_preview, "attrs") else {}
        metadata_tile_exclusion_info = {
            "tile_exclusions_applied": bool(
                attrs.get("tile_exclusions_applied", False)
            ),
            "tile_exclusions_count": int(attrs.get("tile_exclusions_count", 0)),
            "tile_excluded_shortnames": list(attrs.get("tile_excluded_shortnames", [])),
            "tile_flagged_count": int(attrs.get("tile_flagged_count", 0)),
            "tile_flagged_shortnames": list(attrs.get("tile_flagged_shortnames", [])),
            "tile_flagged_classes": list(attrs.get("tile_flagged_classes", [])),
            "tile_flagged_caveats": list(attrs.get("tile_flagged_caveats", [])),
            "tile_exclusion_policy_sha256": attrs.get("tile_exclusion_policy_sha256"),
            "effective_tile_count": int(
                attrs.get("effective_tile_count", len(md_preview))
            ),
        }
    except Exception as exc:
        metadata_crs_info = {
            "source_crs_declared": None,
            "source_crs": None,
            "metric_crs": None,
            "crs_source": None,
            "crs_provenance": None,
            "crs_explicit": None,
            "transform_applied": None,
            "crs_provenance_status": None,
            "crs_explicit_tile_count": None,
            "crs_heuristic_fallback_count": None,
            "crs_missing_explicit_count": None,
            "crs_consistency_issue_count": None,
            "crs_consistency_issue_shortnames": None,
            "crs_unique_explicit_source_crs": None,
            "crs_explicit_source_crs": None,
            "crs_strict_ready": None,
            "error": str(exc),
        }
        preview_record = report_exception(
            exc,
            phase="metadata_preview",
            user_message="Could not resolve metadata CRS/tile-policy preview",
            output_dir=output_dir,
            logger=logger,
            context={
                "metadata_path": metadata_path,
                "tile_exclusion_policy": tile_policy_path,
                "apply_tile_exclusion": apply_tile_exclusion_flag,
                "execution_profile": execution_profile,
            },
        )
        exceptions_log_path = preview_record.get(
            "exceptions_log_path", exceptions_log_path
        )
        exception_records.append(preview_record)
        if execution_profile == "thesis_repro":
            raise

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
        compute_file_sha256(policy_source_path) if policy_source_path.exists() else None
    )
    policy_method = (
        "snapshot_policy" if snapshot_input_path is not None else "config_policy"
    )

    (
        computed_selection_values,
        computed_selection_method,
        computed_selection_source_file,
        computed_selection_source_hash,
    ) = _resolve_computed_selection_values(
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
        require_computed_when_requested: bool = False,
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
            if (
                compute_params
                and require_computed_when_requested
                and computed_value is sentinel
            ):
                raise ValueError(
                    f"{section}.{key} requires computed artifact when --compute-params is enabled."
                )
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
            else (
                computed_selection_source_file
                if n_samples_source == "computed_autoscale_artifact"
                else None
            )
        ),
        source_hash=(
            policy_source_hash
            if n_samples_source in {"config", "snapshot_config"}
            else (
                computed_selection_source_hash
                if n_samples_source == "computed_autoscale_artifact"
                else None
            )
        ),
        compute_args={"explicit_cli": n_samples is not None},
    )

    # Resolve and record critical policy/config parameters centrally.
    resolved_selection_metric = resolve_param(
        section="selection",
        key="metric",
        paths=["selection.metric"],
        parser=lambda v: str(v).strip(),
    )
    resolved_alpha_visual = resolve_param(
        section="selection",
        key="alpha_visual",
        paths=["selection.alpha_visual", "selection.weights.alpha"],
        parser=float,
        computed_value=computed_selection_values.get("alpha_visual", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
        require_computed_when_requested=True,
    )
    resolved_beta_spatial = resolve_param(
        section="selection",
        key="beta_spatial",
        paths=["selection.beta_spatial", "selection.weights.beta"],
        parser=float,
        computed_value=computed_selection_values.get("beta_spatial", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
        require_computed_when_requested=True,
    )
    resolved_gamma_temporal = resolve_param(
        section="selection",
        key="gamma_temporal",
        paths=["selection.gamma_temporal", "selection.weights.gamma"],
        parser=float,
        computed_value=computed_selection_values.get("gamma_temporal", sentinel),
        computed_method=computed_selection_method,
        computed_source_file=computed_selection_source_file,
        computed_source_hash=computed_selection_source_hash,
        prefer_computed_when_requested=True,
        require_computed_when_requested=True,
    )
    resolved_selection_authority = resolve_param(
        section="selection",
        key="selection_authority",
        paths=["selection.selection_authority"],
        parser=_parse_selection_authority,
        default="materialized_csv_primary",
    )
    resolved_objective_authority = resolve_param(
        section="selection",
        key="objective_authority",
        paths=["selection.objective_authority"],
        parser=_parse_objective_authority,
        default="unified_normalized",
    )
    resolved_spatial_constraint = resolve_param(
        section="selection",
        key="spatial_constraint",
        paths=["selection.spatial_constraint"],
        parser=lambda v: _parse_bool(v, label="selection.spatial_constraint"),
    )
    resolved_use_multi_criteria = resolve_param(
        section="selection",
        key="use_multi_criteria",
        paths=["selection.use_multi_criteria"],
        parser=lambda v: _parse_bool(v, label="selection.use_multi_criteria"),
    )
    resolved_use_constraint_integration = resolve_param(
        section="selection",
        key="use_constraint_integration",
        paths=["selection.use_constraint_integration"],
        parser=lambda v: _parse_bool(v, label="selection.use_constraint_integration"),
    )
    resolved_selection_random_state = resolve_param(
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
    resolved_umap_min_dist = resolve_param(
        section="clustering",
        key="umap_min_dist",
        paths=["clustering.umap_min_dist"],
        parser=float,
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

    resolved_feature_model = resolve_param(
        section="feature_extraction",
        key="model",
        paths=["feature_extraction.model"],
        parser=lambda v: str(v).strip().lower(),
    )
    resolved_feature_model_variant = resolve_param(
        section="feature_extraction",
        key="model_variant",
        paths=["feature_extraction.model_variant"],
        parser=lambda v: str(v).strip(),
        default="dinov2_vits14",
        notes="DINOv2 model variant pinned for reproducibility",
    )
    resolved_feature_dinov2_repo = resolve_param(
        section="feature_extraction",
        key="dinov2_repo",
        paths=["feature_extraction.dinov2_repo"],
        parser=lambda v: str(v).strip(),
        default="facebookresearch/dinov2",
        notes="DINOv2 upstream repository",
    )
    resolved_feature_dinov2_ref = resolve_param(
        section="feature_extraction",
        key="dinov2_ref",
        paths=["feature_extraction.dinov2_ref"],
        parser=lambda v: str(v).strip(),
        default="main",
        notes="DINOv2 repository ref (branch/tag/commit)",
    )
    resolved_feature_pooling = resolve_param(
        section="feature_extraction",
        key="pooling",
        paths=["feature_extraction.pooling"],
        parser=_parse_pooling,
        default="cls" if resolved_feature_model == "dinov2" else "global_avg",
        notes="Feature pooling strategy used for embedding extraction",
    )
    resolved_feature_input_size = resolve_param(
        section="feature_extraction",
        key="input_size",
        paths=["feature_extraction.input_size"],
        parser=int,
        default=392 if resolved_feature_model == "dinov2" else 224,
    )
    resolved_feature_resnet_input_size = resolve_param(
        section="feature_extraction",
        key="resnet_input_size",
        paths=["feature_extraction.resnet_input_size"],
        parser=int,
        default=224,
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
        parser=lambda v: (
            [int(v[0]), int(v[1])]
            if isinstance(v, (list, tuple)) and len(v) == 2
            else (
                _parse_positive_int(v, label="feature_extraction.crop_size"),
                _parse_positive_int(v, label="feature_extraction.crop_size"),
            )
        ),
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
    computed_min_distance_km = computed_selection_values.get(
        "min_distance_km", sentinel
    )
    config_min_distance_km = _config_get(parameters, "selection.min_distance_km")
    if compute_params and computed_min_distance_km is not sentinel:
        config_min_distance_km = float(computed_min_distance_km)
        parameters.setdefault("selection", {})[
            "min_distance_km"
        ] = config_min_distance_km
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
        parameters.setdefault("selection", {})[
            "min_distance_km"
        ] = config_min_distance_km
        _record_param_provenance(
            parameters,
            "selection",
            "min_distance_km",
            method="computed_from_metadata",
            source_file=str(metadata_path),
            source_hash=(
                compute_file_sha256(metadata_path) if metadata_path.exists() else None
            ),
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

    if validation_replicate_mode is None:
        cfg_validation_mode = _config_first(
            parameters,
            ["validation.replicate_mode", "selection.validation_replicate_mode"],
        )
        if cfg_validation_mode is None:
            validation_replicate_mode = "bootstrap_candidates"
            _record_param_provenance(
                parameters,
                "selection",
                "validation_replicate_mode",
                method="workflow_default_policy",
                compute_args={"default": "bootstrap_candidates"},
            )
        else:
            validation_replicate_mode = str(cfg_validation_mode).strip().lower()
            _record_param_provenance(
                parameters,
                "selection",
                "validation_replicate_mode",
                method=policy_method,
                source_file=policy_source_file,
                source_hash=policy_source_hash,
                compute_args={"source": "config"},
            )
    else:
        validation_replicate_mode = str(validation_replicate_mode).strip().lower()
        _record_param_provenance(
            parameters,
            "selection",
            "validation_replicate_mode",
            method="explicit_cli",
            compute_args={"source": "cli"},
        )

    if validation_replicate_mode not in {"seed_replay", "bootstrap_candidates"}:
        raise ValueError(
            "validation_replicate_mode must be seed_replay|bootstrap_candidates "
            f"(got {validation_replicate_mode!r})"
        )
    parameters.setdefault("selection", {})[
        "validation_replicate_mode"
    ] = validation_replicate_mode

    if validation_n_bootstrap is None:
        cfg_n_boot = _config_first(
            parameters,
            ["validation.n_bootstrap", "selection.validation_n_bootstrap"],
        )
        validation_n_bootstrap = int(cfg_n_boot) if cfg_n_boot is not None else 200
    else:
        validation_n_bootstrap = int(validation_n_bootstrap)
    if validation_n_bootstrap <= 0:
        raise ValueError("validation_n_bootstrap must be > 0")
    parameters.setdefault("selection", {})["validation_n_bootstrap"] = int(
        validation_n_bootstrap
    )

    if validation_bootstrap_sample_frac is None:
        cfg_frac = _config_first(
            parameters,
            [
                "validation.bootstrap_sample_frac",
                "selection.validation_bootstrap_sample_frac",
            ],
        )
        validation_bootstrap_sample_frac = (
            float(cfg_frac) if cfg_frac is not None else 1.0
        )
    else:
        validation_bootstrap_sample_frac = float(validation_bootstrap_sample_frac)
    if validation_bootstrap_sample_frac <= 0.0:
        raise ValueError("validation_bootstrap_sample_frac must be > 0")
    parameters.setdefault("selection", {})["validation_bootstrap_sample_frac"] = float(
        validation_bootstrap_sample_frac
    )

    # Core+Case contract resolution
    cfg_case_names = _config_first(
        parameters,
        ["selection.case_tile_names", "selection.case_names"],
    )
    if case_names is None and cfg_case_names is None:
        case_names_list: list[str] = []
    elif case_names is None:
        if not isinstance(cfg_case_names, (list, tuple)):
            raise ValueError(
                "selection.case_tile_names must be a list when configured "
                f"(got {type(cfg_case_names)})"
            )
        case_names_list = [str(v) for v in cfg_case_names]
    else:
        case_names_list = [str(v) for v in case_names]

    # Hamburg convenience now maps to case tile by default.
    if hamburg:
        case_names_list.append("Hamburg")
    case_names_list = _dedupe_str_list(case_names_list)
    parameters.setdefault("selection", {})["case_tile_names"] = list(case_names_list)
    _record_param_provenance(
        parameters,
        "selection",
        "case_tile_names",
        method=(
            "explicit_cli"
            if case_names is not None or hamburg
            else (
                policy_method
                if cfg_case_names is not None
                else "workflow_default_policy"
            )
        ),
        source_file=policy_source_file if cfg_case_names is not None else None,
        source_hash=policy_source_hash if cfg_case_names is not None else None,
    )

    if case_exclude_from_core is None:
        cfg_case_exclude = _config_first(
            parameters,
            ["selection.case_exclude_from_core"],
        )
        case_exclude_from_core_flag = (
            _parse_bool(cfg_case_exclude, label="selection.case_exclude_from_core")
            if cfg_case_exclude is not None
            else True
        )
    else:
        case_exclude_from_core_flag = _parse_bool(
            case_exclude_from_core,
            label="case_exclude_from_core",
        )
    parameters.setdefault("selection", {})["case_exclude_from_core"] = bool(
        case_exclude_from_core_flag
    )

    if case_attach_mode is None:
        cfg_attach_mode = _config_first(parameters, ["selection.case_attach_mode"])
        case_attach_mode_resolved = _parse_case_attach_mode(
            cfg_attach_mode if cfg_attach_mode is not None else "append_unique"
        )
    else:
        case_attach_mode_resolved = _parse_case_attach_mode(case_attach_mode)
    parameters.setdefault("selection", {})[
        "case_attach_mode"
    ] = case_attach_mode_resolved

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
        canonical_sampler_artifact = _materialize_sampler_resolution_artifact(
            output_dir=output_dir,
            sampler=resolved_sampler,
            source=sampler_source,
            source_artifact=sampler_artifact_path,
        )
        sampler_artifact_path = str(canonical_sampler_artifact)
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
        source_hash=(
            compute_file_sha256(Path(sampler_artifact_path))
            if sampler_artifact_path and Path(sampler_artifact_path).exists()
            else None
        ),
    )

    resolved_exploration_sampler, exploration_sampler_source = (
        _resolve_exploration_sampler(
            config=parameters,
            resolved_optuna_sampler=resolved_sampler,
        )
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

    pre_names_list = _dedupe_str_list(list(pre_names) if pre_names is not None else [])
    if (
        not case_exclude_from_core_flag
        and hamburg
        and "Hamburg".lower() not in {n.lower() for n in pre_names_list}
    ):
        pre_names_list.append("Hamburg")
    if case_exclude_from_core_flag and case_names_list:
        blocked = {name.lower() for name in case_names_list}
        pre_names_list = [nm for nm in pre_names_list if nm.lower() not in blocked]
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
        source_file=(
            str(metadata_path)
            if n_lhs_source == "computed_adaptive_from_metadata"
            else None
        ),
        source_hash=(
            compute_file_sha256(metadata_path)
            if n_lhs_source == "computed_adaptive_from_metadata"
            and metadata_path.exists()
            else None
        ),
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
    if computed_selection_source_file and Path(computed_selection_source_file).exists():
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

    phase_status: dict[str, str] = {
        "phase1_exploration": "pending",
        "phase2_optimization": "pending",
        "phase3_validation": "pending",
        "phase4_summary": "pending",
        "phase5_handoffs": "pending",
    }
    strict_block_reason: str | None = None
    phase5_extra: dict[str, Any] = {
        "build_handoffs": bool(build_handoffs_flag),
        "handoff_root": handoff_root_resolved,
        "patches_per_tile": int(patches_per_tile_resolved),
        "patch_include_case": bool(patch_include_case_flag),
        "patch_selection_group": "final" if patch_include_case_flag else "core",
        "patch_id_filter_path": patch_id_file_resolved,
        "annotation_plan_dir": None,
        "annotation_dataset_contract_path": None,
        "tile_handoff_dir": None,
        "tile_handoff_manifest_path": None,
        "tile_handoff_selection_count": None,
        "tile_handoff_verify": None,
        "patch_handoff_dir": None,
        "patch_handoff_manifest_path": None,
        "patch_handoff_selection_count": None,
        "patch_handoff_verify": None,
        "patches_total": None,
        "patches_qc_passed": None,
        "patches_qc_rejected": None,
        "phase5_freeze_boundary_verified": False,
        "phase5_freeze_boundary_hashes": None,
        "phase5_freeze_boundary_hashes_post": None,
    }

    if no_auto_continue:
        print("⏹️  Resolution finished; stopping due to --no-auto-continue.")
        phase_status["phase5_handoffs"] = "skipped_resolution_only"
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
                    "n_samples": resolved_n_samples,
                    "n_samples_source": n_samples_source,
                    "parameter_source": parameter_source,
                    "resolved_sampler": resolved_sampler,
                    "resolved_sampler_source": sampler_source,
                    "sampler_artifact_path": sampler_artifact_path,
                    "resolved_exploration_sampler": resolved_exploration_sampler,
                    "resolved_exploration_sampler_source": exploration_sampler_source,
                    "selection_authority": resolved_selection_authority,
                    "objective_authority": resolved_objective_authority,
                    "resolved_snapshot_path": (
                        str(resolved_snapshot_path) if resolved_snapshot_path else None
                    ),
                    "stable_snapshot_path": (
                        str(stable_snapshot_path) if stable_snapshot_path else None
                    ),
                    "resolved_snapshot_sha256": resolved_snapshot_sha256,
                    "snapshot_validation_errors": snapshot_errors,
                    "snapshot_forced": snapshot_forced,
                    "force_override_used": bool(force),
                    "strict_scientific": bool(strict_scientific),
                    "phase_status": phase_status,
                    "strict_block_reason": strict_block_reason,
                    "resolved_n_clusters": resolved_n_clusters,
                    "resolved_batch_size": resolved_batch_size,
                    "resolved_feature_model": resolved_feature_model,
                    "resolved_feature_model_variant": resolved_feature_model_variant,
                    "resolved_feature_pooling": resolved_feature_pooling,
                    "resolved_feature_input_size": resolved_feature_input_size,
                    "resolved_feature_resnet_input_size": resolved_feature_resnet_input_size,
                    "resolved_feature_dinov2_repo": resolved_feature_dinov2_repo,
                    "resolved_feature_dinov2_ref": resolved_feature_dinov2_ref,
                    "resolved_umap_components": resolved_umap_components,
                    "resolved_umap_n_neighbors": resolved_umap_n_neighbors,
                    "resolved_umap_min_dist": resolved_umap_min_dist,
                    "resolved_umap_random_state": resolved_umap_random_state,
                    "resolved_umap_n_jobs": resolved_umap_n_jobs,
                    "computed_selection_method": computed_selection_method,
                    "computed_selection_source_file": computed_selection_source_file,
                    "computed_selection_source_hash": computed_selection_source_hash,
                    "validation_seeds": validation_seeds,
                    "validation_min_distances": validation_min_distances,
                    "validation_replicate_mode": validation_replicate_mode,
                    "validation_n_bootstrap": validation_n_bootstrap,
                    "validation_bootstrap_sample_frac": validation_bootstrap_sample_frac,
                    "pre_selected_names": pre_names_list if pre_names_list else None,
                    "pre_selected_indices": (
                        pre_indices_list if pre_indices_list else None
                    ),
                    "hamburg_shortcut": bool(hamburg),
                    "case_tile_names": case_names_list if case_names_list else None,
                    "case_exclude_from_core": bool(case_exclude_from_core_flag),
                    "case_attach_mode": case_attach_mode_resolved,
                    "tile_exclusion_policy_path": (
                        str(tile_policy_path) if tile_policy_path else None
                    ),
                    "apply_tile_exclusion": bool(apply_tile_exclusion_flag),
                    **phase5_extra,
                    "exceptions_log_path": exceptions_log_path,
                    "exception_records": exception_records,
                    "metadata_crs": metadata_crs_info,
                    **metadata_crs_audit_info,
                    **metadata_tile_exclusion_info,
                },
            )
        except Exception as exc:
            report_exception(
                exc,
                phase="run_metadata_write_resolution_only",
                user_message="Could not write run metadata",
                output_dir=output_dir,
                logger=logger,
                context={"mode": "resolution_only"},
            )
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
        "validation: seeds={}, min_distances={}, mode={}, n_bootstrap={}, bootstrap_sample_frac={}".format(
            validation_seeds,
            validation_min_distances,
            validation_replicate_mode,
            validation_n_bootstrap,
            validation_bootstrap_sample_frac,
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
            (
                resolved_exploration_sampler
                if resolved_exploration_sampler is not None
                else "<unresolved>"
            ),
            exploration_sampler_source,
        )
    )
    print(
        "Preselection: names={}, indices={}".format(
            pre_names_list if pre_names_list else None,
            pre_indices_list if pre_indices_list else None,
        )
    )
    print(
        "Core+Case: case_names={}, case_exclude_from_core={}, case_attach_mode={}".format(
            case_names_list if case_names_list else None,
            bool(case_exclude_from_core_flag),
            case_attach_mode_resolved,
        )
    )
    print(
        "Selection authority: {} | Objective authority: {}".format(
            resolved_selection_authority,
            resolved_objective_authority,
        )
    )
    print("=" * 80)

    all_success = True
    selection_contract_extra: dict[str, Any] = {}

    # Phase 1: Exploration (LHS Sweep)
    if not skip_exploration:
        print("\n" + "=" * 80)
        print("PHASE 1: EXPLORATION (LHS-based Pareto-Front)")
        print("=" * 80)

        if dry_run:
            print(f"[DRY-RUN] Would run: Exploration with n_lhs={n_lhs}")
            phase_status["phase1_exploration"] = "skipped_dry_run"
        else:
            t0 = time.time()
            try:
                print(f"Running Exploration with n_lhs={n_lhs}...")
                run_exploration(
                    n_samples=n_lhs,
                    selection_n_samples=resolved_n_samples,
                    sampler=resolved_exploration_sampler,
                    objective_authority=resolved_objective_authority,
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
                phase_status["phase1_exploration"] = "success"
            except Exception as e:
                record = report_exception(
                    e,
                    phase="phase1_exploration",
                    user_message="FEHLER in Phase 1",
                    output_dir=output_dir,
                    logger=logger,
                    context={
                        "n_lhs": n_lhs,
                        "resolved_n_samples": resolved_n_samples,
                        "metadata_path": metadata_path,
                        "pre_selected_names": pre_names_list,
                        "pre_selected_indices": pre_indices_list,
                    },
                )
                exceptions_log_path = record.get(
                    "exceptions_log_path", exceptions_log_path
                )
                exception_records.append(record)
                all_success = False
                phase_status["phase1_exploration"] = "failed"
                if strict_scientific:
                    strict_block_reason = "strict_scientific=true and Phase 1 failed; subsequent phases blocked"
                elif not skip_optimization and not skip_validation:
                    print("⚠️ Nachfolgende Phasen könnten fehlschlagen")
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 1: Exploration")
        phase_status["phase1_exploration"] = "skipped"

    # Phase 2: Optimization (Optuna)
    if strict_block_reason is not None and strict_scientific:
        print(
            "\n⛔ STRICT MODE: Überspringe Phase 2 aufgrund vorherigem Fehler "
            f"({strict_block_reason})."
        )
        phase_status["phase2_optimization"] = "skipped_due_to_prior_failure"
    elif not skip_optimization:
        print("\n" + "=" * 80)
        print("PHASE 2: OPTIMIZATION (Optuna Bayesian)")
        print("=" * 80)

        if dry_run:
            print(f"[DRY-RUN] Would run: Optuna with n_trials={n_trials}")
            phase_status["phase2_optimization"] = "skipped_dry_run"
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
                    feature_cache_dir=feature_cache_dir,
                    study_name=f"thesis_optuna_{timestamp}",
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 2 erfolgreich (Dauer: {elapsed:.1f}s)")
                phase_status["phase2_optimization"] = "success"
            except Exception as e:
                record = report_exception(
                    e,
                    phase="phase2_optimization",
                    user_message="FEHLER in Phase 2",
                    output_dir=output_dir,
                    logger=logger,
                    context={
                        "n_trials": n_trials,
                        "resolved_n_samples": resolved_n_samples,
                        "metadata_path": metadata_path,
                        "resolved_sampler": resolved_sampler,
                    },
                )
                exceptions_log_path = record.get(
                    "exceptions_log_path", exceptions_log_path
                )
                exception_records.append(record)
                all_success = False
                phase_status["phase2_optimization"] = "failed"
                if strict_scientific:
                    strict_block_reason = "strict_scientific=true and Phase 2 failed; subsequent phases blocked"
                elif not skip_validation:
                    print("⚠️ Validation könnte fehlschlagen")
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 2: Optimization")
        phase_status["phase2_optimization"] = "skipped"

    # Phase 3: Validation
    if strict_block_reason is not None and strict_scientific:
        print(
            "\n⛔ STRICT MODE: Überspringe Phase 3 aufgrund vorherigem Fehler "
            f"({strict_block_reason})."
        )
        phase_status["phase3_validation"] = "skipped_due_to_prior_failure"
    elif not skip_validation:
        print("\n" + "=" * 80)
        print("PHASE 3: VALIDATION (Pareto Candidate Robustness)")
        print("=" * 80)

        if dry_run:
            print("[DRY-RUN] Would run: validation over exploration Pareto candidates")
            phase_status["phase3_validation"] = "skipped_dry_run"
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
                        f"{pareto_csv}. Exploration failed or was not executed."
                    )

                print(f"Running validation for Pareto candidates: {pareto_csv}")
                validate_pareto_candidates(
                    pareto_csv=pareto_csv,
                    min_distances=validation_min_distances,
                    seeds=validation_seeds,
                    output_dir=output_dir / "validation",
                    feature_cache_dir=feature_cache_dir,
                    n_samples=resolved_n_samples,
                    n_clusters=resolved_n_clusters,
                    batch_size=resolved_batch_size,
                    umap_n_components=resolved_umap_components,
                    umap_n_neighbors=resolved_umap_n_neighbors,
                    umap_random_state=resolved_umap_random_state,
                    umap_n_jobs=resolved_umap_n_jobs,
                    pre_selected_names=pre_names_list if pre_names_list else None,
                    pre_selected_indices=pre_indices_list if pre_indices_list else None,
                    replicate_mode=validation_replicate_mode,
                    n_bootstrap=validation_n_bootstrap,
                    bootstrap_sample_frac=validation_bootstrap_sample_frac,
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 3 erfolgreich (Dauer: {elapsed:.1f}s)")
                phase_status["phase3_validation"] = "success"
            except Exception as e:
                record = report_exception(
                    e,
                    phase="phase3_validation",
                    user_message="FEHLER in Phase 3",
                    output_dir=output_dir,
                    logger=logger,
                    context={
                        "validation_min_distances": validation_min_distances,
                        "validation_seeds": validation_seeds,
                        "validation_replicate_mode": validation_replicate_mode,
                        "resolved_n_samples": resolved_n_samples,
                    },
                )
                exceptions_log_path = record.get(
                    "exceptions_log_path", exceptions_log_path
                )
                exception_records.append(record)
                all_success = False
                phase_status["phase3_validation"] = "failed"
    else:
        print("\n⏭️  ÜBERSPRINGE Phase 3: Validation")
        phase_status["phase3_validation"] = "skipped"

    # Phase 4: Summary (always run unless dry-run)
    if not dry_run:
        try:
            selection_contract_extra = _materialize_core_case_artifacts(
                output_dir=output_dir,
                metadata_path=metadata_path,
                config_path=config_path,
                case_names=case_names_list,
                case_exclude_from_core=bool(case_exclude_from_core_flag),
                case_attach_mode=case_attach_mode_resolved,
                selection_authority=resolved_selection_authority,
                objective_authority=resolved_objective_authority,
                selection_params={
                    "n_samples": int(resolved_n_samples),
                    "metric": str(resolved_selection_metric),
                    "alpha_visual": float(resolved_alpha_visual),
                    "beta_spatial": float(resolved_beta_spatial),
                    "gamma_temporal": float(resolved_gamma_temporal),
                    "min_distance_km": float(config_min_distance_km),
                    "spatial_constraint": bool(resolved_spatial_constraint),
                    "use_multi_criteria": bool(resolved_use_multi_criteria),
                    "use_constraint_integration": bool(
                        resolved_use_constraint_integration
                    ),
                    "random_state": int(resolved_selection_random_state),
                },
                resolved_feature_config=parameters.get("feature_extraction", {}),
                pre_selected_names=pre_names_list if pre_names_list else None,
                pre_selected_indices=pre_indices_list if pre_indices_list else None,
                tile_exclusion_policy=tile_policy_path,
                apply_tile_exclusion=apply_tile_exclusion_flag,
            )
            print(
                "✅ Core+Case artifacts written: core={}, case={}, final={}".format(
                    selection_contract_extra.get("selection_core_path"),
                    selection_contract_extra.get("selection_case_path"),
                    selection_contract_extra.get("selection_final_with_cases_path"),
                )
            )
        except Exception as e:
            record = report_exception(
                e,
                phase="core_case_artifact_export",
                user_message="FEHLER beim Core+Case-Artifact-Export",
                output_dir=output_dir,
                logger=logger,
                context={
                    "selection_authority": resolved_selection_authority,
                    "objective_authority": resolved_objective_authority,
                    "case_tile_names": case_names_list,
                    "case_exclude_from_core": bool(case_exclude_from_core_flag),
                    "case_attach_mode": case_attach_mode_resolved,
                    "tile_exclusion_policy": tile_policy_path,
                    "apply_tile_exclusion": apply_tile_exclusion_flag,
                },
            )
            exceptions_log_path = record.get("exceptions_log_path", exceptions_log_path)
            exception_records.append(record)
            all_success = False

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
            phase_status["phase4_summary"] = "success"
        except Exception as e:
            record = report_exception(
                e,
                phase="phase4_summary",
                user_message="FEHLER in Phase 4",
                output_dir=output_dir,
                logger=logger,
                context={
                    "output_dir": output_dir,
                },
            )
            exceptions_log_path = record.get("exceptions_log_path", exceptions_log_path)
            exception_records.append(record)
            all_success = False
            phase_status["phase4_summary"] = "failed"
    else:
        phase_status["phase4_summary"] = "skipped_dry_run"

    # Phase 5: Post-freeze annotation plan + handoff packaging
    if build_handoffs_flag:
        if dry_run:
            phase_status["phase5_handoffs"] = "skipped_dry_run"
        elif not all_success:
            print(
                "\n⏭️  ÜBERSPRINGE Phase 5: Annotation & Handoff "
                "(prior scientific phases did not complete successfully)"
            )
            phase_status["phase5_handoffs"] = "skipped_due_to_prior_failure"
        else:
            print("\n" + "=" * 80)
            print("PHASE 5: ANNOTATION & HANDOFF")
            print("=" * 80)
            t0 = time.time()
            try:
                phase5_extra.update(
                    _run_phase5_annotation_handoffs(
                        output_dir=output_dir,
                        resolved_snapshot_path=resolved_snapshot_path,
                        handoff_root=handoff_root_resolved,
                        patches_per_tile=patches_per_tile_resolved,
                        patch_include_case=patch_include_case_flag,
                        patch_id_file=patch_id_file_resolved or None,
                        tile_exclusion_policy=tile_policy_path,
                    )
                )
                elapsed = time.time() - t0
                print(f"✅ Phase 5 erfolgreich (Dauer: {elapsed:.1f}s)")
                phase_status["phase5_handoffs"] = "success"
            except Exception as e:
                record = report_exception(
                    e,
                    phase="phase5_handoffs",
                    user_message="FEHLER in Phase 5",
                    output_dir=output_dir,
                    logger=logger,
                    context={
                        "handoff_root": handoff_root_resolved,
                        "patches_per_tile": int(patches_per_tile_resolved),
                        "patch_include_case": bool(patch_include_case_flag),
                        "tile_exclusion_policy": tile_policy_path,
                    },
                )
                exceptions_log_path = record.get(
                    "exceptions_log_path", exceptions_log_path
                )
                exception_records.append(record)
                all_success = False
                phase_status["phase5_handoffs"] = "failed"
    else:
        phase_status["phase5_handoffs"] = "skipped"

    metadata_written = False
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
            config_path=config_path,
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
                "selection_authority": resolved_selection_authority,
                "objective_authority": resolved_objective_authority,
                "resolved_snapshot_path": (
                    str(resolved_snapshot_path) if resolved_snapshot_path else None
                ),
                "stable_snapshot_path": (
                    str(stable_snapshot_path) if stable_snapshot_path else None
                ),
                "resolved_snapshot_sha256": resolved_snapshot_sha256,
                "snapshot_validation_errors": snapshot_errors,
                "snapshot_forced": snapshot_forced,
                "force_override_used": bool(force),
                "strict_scientific": bool(strict_scientific),
                "phase_status": phase_status,
                "strict_block_reason": strict_block_reason,
                "n_trials": n_trials,
                "resolved_n_clusters": resolved_n_clusters,
                "resolved_batch_size": resolved_batch_size,
                "resolved_feature_model": resolved_feature_model,
                "resolved_feature_model_variant": resolved_feature_model_variant,
                "resolved_feature_pooling": resolved_feature_pooling,
                "resolved_feature_input_size": resolved_feature_input_size,
                "resolved_feature_resnet_input_size": resolved_feature_resnet_input_size,
                "resolved_feature_dinov2_repo": resolved_feature_dinov2_repo,
                "resolved_feature_dinov2_ref": resolved_feature_dinov2_ref,
                "resolved_umap_components": resolved_umap_components,
                "resolved_umap_n_neighbors": resolved_umap_n_neighbors,
                "resolved_umap_min_dist": resolved_umap_min_dist,
                "resolved_umap_random_state": resolved_umap_random_state,
                "resolved_umap_n_jobs": resolved_umap_n_jobs,
                "computed_selection_method": computed_selection_method,
                "computed_selection_source_file": computed_selection_source_file,
                "computed_selection_source_hash": computed_selection_source_hash,
                "skip_exploration": skip_exploration,
                "skip_optimization": skip_optimization,
                "skip_validation": skip_validation,
                "dry_run": dry_run,
                "cache_mode": str(cache_mode),
                "validation_seeds": validation_seeds,
                "validation_min_distances": validation_min_distances,
                "validation_replicate_mode": validation_replicate_mode,
                "validation_n_bootstrap": validation_n_bootstrap,
                "validation_bootstrap_sample_frac": validation_bootstrap_sample_frac,
                "pre_selected_names": pre_names_list if pre_names_list else None,
                "pre_selected_indices": pre_indices_list if pre_indices_list else None,
                "hamburg_shortcut": bool(hamburg),
                "case_tile_names": case_names_list if case_names_list else None,
                "case_exclude_from_core": bool(case_exclude_from_core_flag),
                "case_attach_mode": case_attach_mode_resolved,
                "tile_exclusion_policy_path": (
                    str(tile_policy_path) if tile_policy_path else None
                ),
                "apply_tile_exclusion": bool(apply_tile_exclusion_flag),
                **phase5_extra,
                "exceptions_log_path": exceptions_log_path,
                "exception_records": exception_records,
                "metadata_crs": metadata_crs_info,
                **metadata_crs_audit_info,
                **metadata_tile_exclusion_info,
                **selection_contract_extra,
            },
        )
        metadata_written = True
    except Exception as exc:
        report_exception(
            exc,
            phase="run_metadata_write_final",
            user_message="Could not write run metadata",
            output_dir=output_dir,
            logger=logger,
            context={"mode": "final"},
        )

    if not dry_run and metadata_written:
        try:
            generate_thesis_final_report(
                output_dir=output_dir,
                timestamp=timestamp,
            )
        except Exception as exc:
            record = report_exception(
                exc,
                phase="final_report_refresh",
                user_message="Could not refresh final thesis report after Phase 5",
                output_dir=output_dir,
                logger=logger,
                context={"output_dir": output_dir},
            )
            exceptions_log_path = record.get("exceptions_log_path", exceptions_log_path)
            exception_records.append(record)
            all_success = False
            phase_status["phase4_summary"] = "failed_post_phase5_refresh"
            try:
                write_run_metadata(
                    output_dir=output_dir,
                    execution_profile=execution_profile,
                    seed=seed,
                    command=sys.argv,
                    config_path=config_path,
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
                        "selection_authority": resolved_selection_authority,
                        "objective_authority": resolved_objective_authority,
                        "resolved_snapshot_path": (
                            str(resolved_snapshot_path)
                            if resolved_snapshot_path
                            else None
                        ),
                        "stable_snapshot_path": (
                            str(stable_snapshot_path) if stable_snapshot_path else None
                        ),
                        "resolved_snapshot_sha256": resolved_snapshot_sha256,
                        "snapshot_validation_errors": snapshot_errors,
                        "snapshot_forced": snapshot_forced,
                        "force_override_used": bool(force),
                        "strict_scientific": bool(strict_scientific),
                        "phase_status": phase_status,
                        "strict_block_reason": strict_block_reason,
                        "n_trials": n_trials,
                        "resolved_n_clusters": resolved_n_clusters,
                        "resolved_batch_size": resolved_batch_size,
                        "resolved_feature_model": resolved_feature_model,
                        "resolved_feature_model_variant": resolved_feature_model_variant,
                        "resolved_feature_pooling": resolved_feature_pooling,
                        "resolved_feature_input_size": resolved_feature_input_size,
                        "resolved_feature_resnet_input_size": resolved_feature_resnet_input_size,
                        "resolved_feature_dinov2_repo": resolved_feature_dinov2_repo,
                        "resolved_feature_dinov2_ref": resolved_feature_dinov2_ref,
                        "resolved_umap_components": resolved_umap_components,
                        "resolved_umap_n_neighbors": resolved_umap_n_neighbors,
                        "resolved_umap_min_dist": resolved_umap_min_dist,
                        "resolved_umap_random_state": resolved_umap_random_state,
                        "resolved_umap_n_jobs": resolved_umap_n_jobs,
                        "computed_selection_method": computed_selection_method,
                        "computed_selection_source_file": computed_selection_source_file,
                        "computed_selection_source_hash": computed_selection_source_hash,
                        "skip_exploration": skip_exploration,
                        "skip_optimization": skip_optimization,
                        "skip_validation": skip_validation,
                        "dry_run": dry_run,
                        "cache_mode": str(cache_mode),
                        "validation_seeds": validation_seeds,
                        "validation_min_distances": validation_min_distances,
                        "validation_replicate_mode": validation_replicate_mode,
                        "validation_n_bootstrap": validation_n_bootstrap,
                        "validation_bootstrap_sample_frac": validation_bootstrap_sample_frac,
                        "pre_selected_names": (
                            pre_names_list if pre_names_list else None
                        ),
                        "pre_selected_indices": (
                            pre_indices_list if pre_indices_list else None
                        ),
                        "hamburg_shortcut": bool(hamburg),
                        "case_tile_names": case_names_list if case_names_list else None,
                        "case_exclude_from_core": bool(case_exclude_from_core_flag),
                        "case_attach_mode": case_attach_mode_resolved,
                        "tile_exclusion_policy_path": (
                            str(tile_policy_path) if tile_policy_path else None
                        ),
                        "apply_tile_exclusion": bool(apply_tile_exclusion_flag),
                        **phase5_extra,
                        "exceptions_log_path": exceptions_log_path,
                        "exception_records": exception_records,
                        "metadata_crs": metadata_crs_info,
                        **metadata_crs_audit_info,
                        **metadata_tile_exclusion_info,
                        **selection_contract_extra,
                    },
                )
            except Exception as metadata_exc:
                report_exception(
                    metadata_exc,
                    phase="run_metadata_write_after_report_refresh_failure",
                    user_message="Could not update run metadata after report refresh failure",
                    output_dir=output_dir,
                    logger=logger,
                    context={"mode": "post_phase5_refresh_failure"},
                )

    print("\n" + "=" * 80)
    if all_success:
        print("✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN")
    else:
        print("❌ PIPELINE MIT FEHLERN ABGESCHLOSSEN")
    print("=" * 80)

    return all_success


@cli_command(
    "thesis-pipeline",
    help="Run complete thesis optimization pipeline (4 scientific phases + optional Phase 5 handoff bundle)",
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
            "default": 370,
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
        "config_path": {
            "type": str,
            "default": "config/pipeline_config.yaml",
            "help": "Configuration path used for parameter resolution and feature extraction",
        },
        "cache_mode": {
            "type": str,
            "default": "read_write",
            "choices": ["off", "read_only", "write_only", "read_write"],
            "help": "Feature cache mode for downstream workflows",
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
        "strict_scientific": {
            "type": bool,
            "default": True,
            "help": "Fail-fast after phase errors and enforce strict scientific sequencing",
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
            "help": "Convenience shortcut: add Hamburg to case tiles (core+case contract)",
        },
        "case_names": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Additional case tiles appended after core selection",
        },
        "case_exclude_from_core": {
            "type": str,
            "default": None,
            "choices": ["true", "false"],
            "help": "Exclude case tiles from core selection (default: true)",
        },
        "case_attach_mode": {
            "type": str,
            "default": None,
            "choices": ["append_unique", "append_all"],
            "help": "How case tiles are attached after core selection",
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
        "validation_replicate_mode": {
            "type": str,
            "default": None,
            "choices": ["seed_replay", "bootstrap_candidates"],
            "help": "Validation replicate mode (inferential default: bootstrap_candidates)",
        },
        "validation_n_bootstrap": {
            "type": int,
            "default": None,
            "help": "Number of bootstrap replicates for validation",
        },
        "validation_bootstrap_sample_frac": {
            "type": float,
            "default": None,
            "help": "Bootstrap candidate sampling fraction",
        },
        "tile_exclusion_policy": {
            "type": str,
            "default": None,
            "help": "Explicit tile exclusion/flagging policy path",
        },
        "apply_tile_exclusion": {
            "type": str,
            "default": None,
            "choices": ["true", "false"],
            "help": "Explicitly enable or disable tile exclusion policy application",
        },
        "build_handoffs": {
            "type": bool,
            "action": "store_true",
            "help": "Run optional post-freeze Phase 5 tile/patch handoff bundle",
        },
        "patches_per_tile": {
            "type": int,
            "default": 2,
            "help": "Patches per tile for integrated annotation-plan build",
        },
        "patch_include_case": {
            "type": str,
            "default": "false",
            "choices": ["true", "false"],
            "help": "Include case tiles in the integrated patch plan/handoff",
        },
        "patch_id_file": {
            "type": str,
            "default": "",
            "help": "Optional plain-text patch-id allowlist for a filtered Phase-5 patch handoff",
        },
        "handoff_root": {
            "type": str,
            "default": "handoff",
            "help": "Root directory for integrated tile/patch handoff bundles",
        },
    },
)
def main(
    n_lhs: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_trials: int = 370,
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
    config_path: str = "config/pipeline_config.yaml",
    cache_mode: str = "read_write",
    execution_profile: str = "default",
    seed: int = 42,
    strict_scientific: bool = True,
    pre_names: Optional[list[str]] = None,
    pre_indices: Optional[list[int]] = None,
    hamburg: bool = False,
    case_names: Optional[list[str]] = None,
    case_exclude_from_core: Optional[str] = None,
    case_attach_mode: Optional[str] = None,
    validation_seeds: Optional[list[int]] = None,
    validation_min_distances: Optional[list[float]] = None,
    validation_replicate_mode: Optional[str] = None,
    validation_n_bootstrap: Optional[int] = None,
    validation_bootstrap_sample_frac: Optional[float] = None,
    tile_exclusion_policy: Optional[str] = None,
    apply_tile_exclusion: Optional[str] = None,
    build_handoffs: bool = False,
    patches_per_tile: int = 2,
    patch_include_case: str = "false",
    patch_id_file: str = "",
    handoff_root: str = "handoff",
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
        config_path=Path(config_path),
        cache_mode=cache_mode,
        execution_profile=execution_profile,
        seed=seed,
        strict_scientific=strict_scientific,
        pre_names=pre_names,
        pre_indices=pre_indices,
        hamburg=hamburg,
        case_names=case_names,
        case_exclude_from_core=(
            _parse_bool(case_exclude_from_core, label="case_exclude_from_core")
            if case_exclude_from_core is not None
            else None
        ),
        case_attach_mode=case_attach_mode,
        validation_seeds=validation_seeds,
        validation_min_distances=validation_min_distances,
        validation_replicate_mode=validation_replicate_mode,
        validation_n_bootstrap=validation_n_bootstrap,
        validation_bootstrap_sample_frac=validation_bootstrap_sample_frac,
        tile_exclusion_policy=(
            Path(tile_exclusion_policy) if tile_exclusion_policy else None
        ),
        apply_tile_exclusion=(
            _parse_bool(
                apply_tile_exclusion,
                label="apply_tile_exclusion",
            )
            if apply_tile_exclusion is not None
            else None
        ),
        build_handoffs=bool(build_handoffs),
        patches_per_tile=patches_per_tile,
        patch_include_case=_parse_bool(
            patch_include_case,
            label="patch_include_case",
        ),
        patch_id_file=patch_id_file or None,
        handoff_root=handoff_root,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
