from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_script(module_path: str | Path, argv: Sequence[str]) -> int:
    """Run an existing repo script in a subprocess.

    We use a subprocess to preserve the current behavior and dependency import
    order of scripts, while we migrate logic gradually into the library.
    """

    script = str(module_path)
    cmd = [sys.executable, script, *list(argv)]
    env = os.environ.copy()
    # Ensure repo root is on PYTHONPATH for script imports.
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call(cmd, env=env)
