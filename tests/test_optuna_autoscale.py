import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Skip tests that require optional heavy dependencies when not installed
if importlib.util.find_spec("optuna") is None:
    pytest.skip("optuna not installed in this environment", allow_module_level=True)

OUT = Path("outputs")


def test_optuna_autoscale_smoke(tmp_path):
    # Run with tiny synthetic setup: 2 stages (5, 10) and small trials to make it fast
    cmd = [
        sys.executable,
        "scripts/optuna_autoscale.py",
        "--n-trials",
        "3",
        "3",
        "--stages",
        "5",
        "10",
        "--n-candidates",
        "50",
        "--dim",
        "16",
        "--seed",
        "1",
    ]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == 0, f"optuna_autoscale failed: {result.stdout}"

    date = datetime.now().strftime("%Y%m%d")
    summary = OUT / f"optuna_autoscale_summary_{date}.csv"
    best = OUT / f"optuna_autoscale_best_{date}.json"
    report = OUT / f"optuna_autoscale_report_{date}.md"
    assert summary.exists(), f"Summary {summary} not created"
    assert best.exists(), f"Best json {best} not created"
    assert report.exists(), f"Report {report} not created"
