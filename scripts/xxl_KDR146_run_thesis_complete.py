#!/usr/bin/env python3
"""Deprecated shim (kept for compatibility).

This module is a lightweight shim that forwards invocations to
`scripts/xxl_KDR146_run_thesis_complete_modern.py`. Prefer calling
`xxl_KDR146_run_thesis_complete_modern.py` directly in new scripts
and documentation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
modern = ROOT / "scripts" / "xxl_KDR146_run_thesis_complete_modern.py"
if not modern.exists():
    print("ERROR: modern orchestrator not found: scripts/xxl_KDR146_run_thesis_complete_modern.py", file=sys.stderr)
    sys.exit(2)

# Forward arguments to the modern script
cmd = [sys.executable, str(modern)] + sys.argv[1:]
# Ensure repo root is on PYTHONPATH for direct invocation
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)
rc = subprocess.run(cmd, env=env)
sys.exit(rc.returncode)
