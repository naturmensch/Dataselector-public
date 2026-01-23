import subprocess
import sys


def test_run_coarse_sweep_deprecated_message():
    res = subprocess.run(
        [sys.executable, "scripts/run_coarse_sweep.py"], capture_output=True, text=True
    )
    out = (res.stdout or "") + (res.stderr or "")
    assert "DEPRECATED" in out or "deprecated" in out.lower()
    assert res.returncode == 0
