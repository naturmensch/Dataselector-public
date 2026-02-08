"""Run metadata writer for reproducible workflow execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _git_commit_sha() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(
    output_dir: str | Path,
    execution_profile: str,
    seed: int,
    command: list[str] | None = None,
    config_path: str | Path | None = None,
    runtime_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write run metadata JSON into output_dir.

    Parameters are intentionally generic so workflows can log context without
    duplicating metadata logic.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = command if command is not None else sys.argv

    tracked_env_keys = [
        "DATASELECTOR_EXECUTION_PROFILE",
        "DATASELECTOR_EXECUTION_SEED",
        "RUN_FULL_INTEGRATION",
        "DATASELECTOR_IMAGE_DIR",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "PYTHONHASHSEED",
    ]

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_commit_sha(),
        "execution_profile": execution_profile,
        "seed": int(seed),
        "command": list(cmd),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "runtime_state": runtime_state or {},
        "environment": {
            k: os.environ.get(k) for k in tracked_env_keys if k in os.environ
        },
    }

    if config_path:
        cfg = Path(config_path)
        metadata["config"] = {
            "path": str(cfg),
            "exists": cfg.exists(),
            "sha256": _sha256(cfg) if cfg.exists() else None,
        }

    if extra:
        metadata["extra"] = extra

    metadata_path = out / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata_path
