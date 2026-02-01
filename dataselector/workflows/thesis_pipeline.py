from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the shell-based thesis pipeline orchestrator via subprocess.

    Canonical usage:
        python -m dataselector thesis-pipeline -- <args>
    """

    argv = list(argv or [])

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_complete_thesis_pipeline.sh"

    env = os.environ.copy()
    # Ensure repo root is on PYTHONPATH (some child scripts depend on it)
    env["PYTHONPATH"] = str(repo_root) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    cmd = ["bash", str(script), *argv]
    return subprocess.call(cmd, env=env, cwd=str(repo_root))
