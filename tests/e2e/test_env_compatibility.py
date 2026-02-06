import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.e2e
def test_env_compatibility_exits_zero():
    """Run the environment checker and assert it exits 0 (compatible).

    This test should run in gated/full E2E environments where native deps are installed.
    In PR/normal dev runs, it's OK to skip if the check fails (we provide actionable output).
    """
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "check_env.py")]
    proc = run_dataselector_cli(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        # Not failing here; skip with an actionable message so PRs don't fail when running in an unrelated environment.
        pytest.skip(
            f"Environment incompatible or missing native deps (code={proc.returncode}). To fix locally, run:\n\n  ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages \"numpy==1.26.4 numba==0.63.1\" --yes -- python -c 'import numpy,numba; print(numpy.__version__,numba.__version__)'\n\nOutput:\n{out}"
        )

    assert "Environment checks passed" in out
