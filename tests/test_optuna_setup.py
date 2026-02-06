import os
import subprocess
import sys
from pathlib import Path

OUT = Path("outputs")


def test_optuna_command_runs(tmp_path):
    # Run package command with smoke settings
    cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "optuna-optimize",
        "--smoke",
        "--n-trials",
        "1",
        "--n-candidates",
        "20",
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
    assert result.returncode == 0, f"optuna command failed: {result.stdout}"
    assert (OUT / "optuna_results.csv").exists()
