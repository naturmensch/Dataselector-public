"""XXL monitor compatibility wrapper (NOT a CLI command).

This module intentionally avoids legacy script entrypoints. It delegates to the
canonical package CLI surface.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _detect_n_candidates(root: str | Path | None = None) -> int:
    """Detect candidate count from env var or data/new_all_tiles.csv."""
    import os

    env_value = os.environ.get("DATASET_N_CANDIDATES")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass

    base = Path(root) if root is not None else Path.cwd()
    csv_path = base / "data" / "new_all_tiles.csv"
    if not csv_path.exists():
        return 676

    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = sum(1 for _ in handle)
    except OSError:
        return 676

    return max(lines - 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Delegate to canonical CLI workflow command."""
    argv = list(argv or [])
    proc = subprocess.run([sys.executable, "-m", "dataselector", "xxl", *argv])
    return int(proc.returncode)
