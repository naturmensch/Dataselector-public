import subprocess
import sys


def test_coarse_sweep_command_is_removed():
    res = subprocess.run(
        [sys.executable, "-m", "dataselector", "coarse-sweep"],
        capture_output=True,
        text=True,
    )
    out = (res.stdout or "") + (res.stderr or "")
    assert res.returncode != 0
    assert "invalid choice" in out.lower() or "unrecognized arguments" in out.lower()
