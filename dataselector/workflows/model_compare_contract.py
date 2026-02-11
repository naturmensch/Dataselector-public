"""Contracts for fair model comparison on identical spatial splits."""

from __future__ import annotations

from typing import Any


def assert_same_split_manifest_sha256(run_metadata_payloads: list[dict[str, Any]]) -> str:
    """Require identical split manifest hash across model runs.

    Returns the common manifest hash if valid.
    """
    hashes = []
    for payload in run_metadata_payloads:
        extra = payload.get("extra", payload)
        sha = extra.get("split_manifest_sha256")
        if not sha:
            raise ValueError("Missing split_manifest_sha256 in run metadata payload.")
        hashes.append(str(sha))
    unique = sorted(set(hashes))
    if len(unique) != 1:
        raise ValueError(
            "Model comparison requires identical split_manifest_sha256 across runs: "
            f"{unique}"
        )
    return unique[0]
