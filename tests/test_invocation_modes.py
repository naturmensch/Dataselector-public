import pytest
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

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
    # Provide log context if failure
    if res.returncode != 0:
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
    assert res.returncode == 0


def test_run_adaptive_pipeline_both_invocations(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Module invocation (dry-run, non-interactive)
    cmd_mod = [
        sys.executable,
        "-m",
        "scripts.run_adaptive_pipeline",
        "--dry-run",
        "--yes",
        "--n-lhs",
        "1",
        "--n-trials",
        "1",
        "--n-boot",
        "1",
        "--seed",
        "1",
    ]
    run_cmd(cmd_mod, cwd=tmp_path, env=env)

    # Direct script invocation
    cmd_file = [
        sys.executable,
        str(ROOT / "scripts" / "run_adaptive_pipeline.py"),
        "--dry-run",
        "--yes",
        "--n-lhs",
        "1",
        "--n-trials",
        "1",
        "--n-boot",
        "1",
        "--seed",
        "1",
    ]
    run_cmd(cmd_file, cwd=tmp_path, env=env)


def test_optuna_optimize_both_invocations(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Module invocation: small run
    cmd_mod = [
        sys.executable,
        "-m",
        "scripts.optuna_optimize",
        "--n-trials",
        "2",
        "--n-candidates",
        "4",
        "--dim",
        "4",
        "--n-samples",
        "1",
        "--exp-name",
        "invocation_test",
        "--seed",
        "1",
    ]
    run_cmd(cmd_mod, cwd=tmp_path, env=env)

    # Direct script invocation
    cmd_file = [
        sys.executable,
        str(ROOT / "scripts" / "optuna_optimize.py"),
        "--n-trials",
        "1",
        "--n-candidates",
        "4",
        "--dim",
        "4",
        "--n-samples",
        "1",
        "--exp-name",
        "invocation_test2",
    ]
    run_cmd(cmd_file, cwd=tmp_path, env=env)


def test_import_trials_csv_to_optuna_both_invocations(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Create CSV
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

    # Module invocation
    cmd_mod = [
        sys.executable,
        "-m",
        "scripts.import_trials_csv_to_optuna",
        "--csv",
        str(csv_path),
        "--storage",
        storage,
        "--study-name",
        "inv_test",
    ]
    run_cmd(cmd_mod, cwd=tmp_path, env=env)

    # Direct script invocation
    cmd_file = [
        sys.executable,
        str(ROOT / "scripts" / "import_trials_csv_to_optuna.py"),
        "--csv",
        str(csv_path),
        "--storage",
        storage,
        "--study-name",
        "inv_test2",
    ]
    run_cmd(cmd_file, cwd=tmp_path, env=env)

    # Verify DBs exist
    assert db_path.exists()
