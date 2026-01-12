import os
import subprocess
import sys
from pathlib import Path

import pytest

OUT = Path("outputs")


def test_optuna_script_runs(tmp_path):
    try:
        pass  # type: ignore
    except Exception:
        pytest.skip("optuna not installed in test environment")

    # Run optuna script with minimal trials to smoke-test
    cmd = [
        sys.executable,
        "scripts/optuna_optimize.py",
        "--n-trials",
        "2",
        "--n-candidates",
        "100",
        "--dim",
        "32",
        "--n-samples",
        "5",
    ]

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Check return code and output file
    assert result.returncode == 0, f"optuna script failed: {result.stdout}"
    assert (OUT / "optuna_results.csv").exists()
