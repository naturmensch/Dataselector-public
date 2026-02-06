import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "xxl_KDR146_run_thesis_complete_modern.py"


def test_xxl_modern_help():
    """Modern orchestrator should return zero with --help (usage printed)"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower() or "usage" in proc.stderr.lower()
