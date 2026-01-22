import pytest

pytest.importorskip("numba", exc_type=ImportError)
pytestmark = pytest.mark.integration

import os
import subprocess
import sys


def test_help_shows_no_hard_default_34():
    """Ensure CLI help doesn't advertise a hardcoded default of 34 for --n-samples."""
    env = os.environ.copy()
    # ensure package imports from repo root work during subprocess execution
    env.setdefault("PYTHONPATH", os.getcwd())
    res = subprocess.run(
        [sys.executable, "scripts/run_adaptive_pipeline.py", "--help"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = res.stdout + res.stderr
    assert "default: 34" not in out.lower(), "Help text still mentions 'default: 34'"
    assert "--n-samples" in out
    assert "overrides adaptive" in out.lower() or "overrides" in out.lower()
