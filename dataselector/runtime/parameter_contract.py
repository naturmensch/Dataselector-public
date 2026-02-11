"""Validation helpers for scientific parameter resolution contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_parameter_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Parameter contract not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid parameter contract format: {path}")
    params = payload.get("parameters")
    if not isinstance(params, dict):
        raise ValueError(f"Parameter contract missing 'parameters' mapping: {path}")
    return payload


def _get_nested(mapping: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _split_section_key(dotted_path: str) -> tuple[str, str]:
    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid parameter path: {dotted_path}")
    return parts[0], ".".join(parts[1:])


def validate_snapshot_against_contract(
    *,
    snapshot: dict[str, Any],
    contract: dict[str, Any],
    run_root: Path,
    repo_root: Path | None = None,
    evidence_scope: str = "run_dir",
) -> list[str]:
    errors: list[str] = []
    repo_base = repo_root or run_root
    if evidence_scope not in {"run_dir", "repo_root"}:
        raise ValueError(
            f"Unknown evidence_scope '{evidence_scope}'. Expected run_dir|repo_root."
        )

    params = snapshot.get("parameters")
    if not isinstance(params, dict):
        return ["Snapshot missing 'parameters' mapping"]

    contract_params = contract.get("parameters", {})
    for dotted, rule in contract_params.items():
        if not isinstance(rule, dict):
            errors.append(f"Contract rule for '{dotted}' must be a mapping")
            continue

        section, key = _split_section_key(str(dotted))
        section_map = params.get(section)
        if not isinstance(section_map, dict):
            errors.append(f"Snapshot missing section '{section}' for '{dotted}'")
            continue

        value = _get_nested(params, dotted)
        if value is None:
            errors.append(f"Snapshot missing value for '{dotted}'")
            continue

        provenance = section_map.get("_provenance", {})
        if not isinstance(provenance, dict):
            errors.append(f"Snapshot missing provenance mapping for section '{section}'")
            continue

        entry = provenance.get(key)
        if not isinstance(entry, dict):
            errors.append(f"Snapshot missing provenance entry for '{dotted}'")
            continue

        method = str(entry.get("method", "")).strip()
        allowed = rule.get("allowed_methods", [])
        if isinstance(allowed, list) and allowed:
            # Allow prefix matches for artifact:* methods.
            if not any(
                method == item or (item.endswith(":*") and method.startswith(item[:-1]))
                for item in map(str, allowed)
            ):
                errors.append(
                    f"'{dotted}' provenance method '{method}' not in allowed_methods {allowed}"
                )

        required_evidence = rule.get("required_evidence")
        if isinstance(required_evidence, str) and required_evidence.strip():
            evidence = required_evidence.strip()
            # Special symbolic evidence key.
            if evidence == "resolved_optuna_sampler":
                # exploration sampler must have compute args linking optuna sampler
                compute_args = entry.get("compute_args", {})
                if not isinstance(compute_args, dict) or not compute_args.get(
                    "resolved_optuna_sampler"
                ):
                    errors.append(
                        f"'{dotted}' requires compute_args.resolved_optuna_sampler evidence"
                    )
            else:
                if evidence.startswith("repo:"):
                    ev_path = repo_base / evidence[len("repo:") :]
                elif evidence_scope == "repo_root":
                    ev_path = repo_base / evidence
                else:
                    ev_path = run_root / evidence
                if not ev_path.exists():
                    # Also allow run-relative evidence if source_file exists and matches suffix.
                    source_file = str(entry.get("source_file", "")).strip()
                    source_path = Path(source_file) if source_file else None
                    source_exists = bool(source_path and source_path.exists())
                    if not source_file or not source_file.endswith(evidence):
                        errors.append(
                            f"'{dotted}' requires evidence '{evidence}' (missing at {ev_path})"
                        )
                    elif not source_exists:
                        errors.append(
                            f"'{dotted}' provenance source_file '{source_file}' does not exist"
                        )

    return errors
