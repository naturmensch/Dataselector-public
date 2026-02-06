import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


@pytest.fixture(autouse=True)
def skip_if_no_optuna():
    pytest.importorskip("optuna", exc_type=ImportError)


def run_cmd(cmd, cwd, env):
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
    assert res.returncode == 0


def test_adaptive_pipeline_package_invocation(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "adaptive-pipeline",
        "--dry-run",
        "--n-lhs",
        "1",
        "--n-trials",
        "1",
        "--n-boot",
        "1",
        "--seed",
        "1",
        "--skip-optuna",
    ]
    run_cmd(cmd, cwd=tmp_path, env=env)


def test_optuna_import_package_invocation(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    csv_path = tmp_path / "trials.csv"
    df = pd.DataFrame(
        {
            "number": [0, 1],
            "value": [0.5, 0.6],
            "state": ["COMPLETE", "COMPLETE"],
            "a": [0.1, 0.2],
            "b": [0.2, 0.3],
            "c": [0.7, 0.5],
            "min_distance_km": [10, 20],
            "n_samples": [5, 5],
        }
    )
    df.to_csv(csv_path, index=False)

    db_path = tmp_path / "study.db"
    storage = f"sqlite:///{db_path}"

    cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "optuna-import",
        "--csv",
        str(csv_path),
        "--storage",
        storage,
        "--study-name",
        "inv_test",
    ]
    run_cmd(cmd, cwd=tmp_path, env=env)
    assert db_path.exists()
