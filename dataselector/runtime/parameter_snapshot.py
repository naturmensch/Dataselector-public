"""Helpers for parameter snapshotting with hashes and provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_parameters_hash(parameters: Dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(parameters).encode("utf-8"))


def build_snapshot(
    parameters: Dict[str, Any],
    provenance: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    return {
        "parameters": parameters,
        "provenance": provenance or {},
        "metadata": metadata
        or {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "hashes": {
            "parameters_hash": compute_parameters_hash(parameters),
            "snapshot_content_sha256": None,
        },
        "notes": notes or "",
    }


def write_snapshot(snapshot: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provisional = dict(snapshot)
    hashes = dict(provisional.get("hashes", {}))
    hashes["snapshot_content_sha256"] = None
    provisional["hashes"] = hashes

    payload = yaml.safe_dump(provisional, sort_keys=True, allow_unicode=False)
    content_hash = _sha256_bytes(payload.encode("utf-8"))
    snapshot["hashes"]["snapshot_content_sha256"] = content_hash

    final_payload = yaml.safe_dump(snapshot, sort_keys=True, allow_unicode=False)
    output_path.write_text(final_payload, encoding="utf-8")
    return snapshot


def _snapshot_content_hash(snapshot: Dict[str, Any]) -> str:
    provisional = dict(snapshot)
    hashes = dict(provisional.get("hashes", {}))
    hashes["snapshot_content_sha256"] = None
    provisional["hashes"] = hashes
    payload = yaml.safe_dump(provisional, sort_keys=True, allow_unicode=False)
    return _sha256_bytes(payload.encode("utf-8"))


def validate_snapshot(snapshot: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    parameters = snapshot.get("parameters")
    if parameters is None:
        errors.append("Missing 'parameters' section.")
    else:
        expected = snapshot.get("hashes", {}).get("parameters_hash")
        if expected:
            actual = compute_parameters_hash(parameters)
            if actual != expected:
                errors.append(
                    "parameters_hash mismatch: expected {} got {}.".format(
                        expected, actual
                    )
                )

    expected_content = snapshot.get("hashes", {}).get("snapshot_content_sha256")
    if expected_content:
        actual_content = _snapshot_content_hash(snapshot)
        if actual_content != expected_content:
            errors.append(
                "snapshot_content_sha256 mismatch: expected {} got {}.".format(
                    expected_content, actual_content
                )
            )

    provenance = snapshot.get("provenance", {})
    source_files = provenance.get("source_files", {})
    if isinstance(source_files, dict):
        for key, info in source_files.items():
            if not isinstance(info, dict):
                continue
            path_value = info.get("path") or info.get("file")
            if not path_value:
                continue
            fpath = Path(path_value)
            if not fpath.exists():
                errors.append(f"source_files.{key} missing: {fpath}")
                continue
            expected_hash = info.get("sha256")
            if expected_hash:
                actual_hash = compute_file_sha256(fpath)
                if actual_hash != expected_hash:
                    errors.append(
                        "source_files.{} hash mismatch: expected {} got {}.".format(
                            key, expected_hash, actual_hash
                        )
                    )

    # Per-parameter provenance validation (additive contract):
    # <section>._provenance.<param> -> {method, source_file, source_hash, ...}
    if isinstance(parameters, dict):
        for section_name, section in parameters.items():
            if not isinstance(section, dict):
                continue
            section_prov = section.get("_provenance")
            if not isinstance(section_prov, dict):
                continue

            for param_name, entry in section_prov.items():
                if not isinstance(entry, dict):
                    continue
                method = str(entry.get("method", "")).strip().lower()
                source_file = entry.get("source_file")
                source_hash = entry.get("source_hash")
                key = f"{section_name}._provenance.{param_name}"

                # Manual/policy-tagged values may intentionally omit file hashes.
                if method in {"manual", "policy", "config_policy", "snapshot_policy"}:
                    continue

                if not source_file:
                    if source_hash:
                        errors.append(f"{key} has source_hash but missing source_file.")
                    continue

                p = Path(str(source_file))
                if not p.exists():
                    errors.append(f"{key} source_file missing: {p}")
                    continue

                if source_hash:
                    actual = compute_file_sha256(p)
                    if actual != source_hash:
                        errors.append(
                            "{} hash mismatch: expected {} got {}.".format(
                                key, source_hash, actual
                            )
                        )

    return errors


def load_snapshot(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_snapshot_file(path: Path) -> list[str]:
    snapshot = load_snapshot(path)
    return validate_snapshot(snapshot)
