# Global pytest configuration and fixtures
import os
import subprocess
import sys

import numpy as np
import pytest

# Ensure repository root is on sys.path so `src` can be imported in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_sessionstart(session):
    """Enforce running tests inside the project's `dataselector` conda env.

    The check considers the `DATASELECTOR_ENV_NAME` env var (if set) or defaults
    to 'dataselector'. If the active conda env (CONDA_DEFAULT_ENV) does not match,
    the test run aborts with an instructive message describing how to create
    and activate the environment.

    A developer can set SKIP_ENV_CHECK=1 or DATASELECTOR_IGNORE_ENV_CHECK=1 to bypass this guard.
    """
    env_name = os.environ.get("DATASELECTOR_ENV_NAME", "dataselector")
    current = os.environ.get("CONDA_DEFAULT_ENV")
    
    # Allow bypass for testing
    if os.environ.get("SKIP_ENV_CHECK") == "1" or os.environ.get("DATASELECTOR_IGNORE_ENV_CHECK") == "1":
        return

    if current != env_name:
        msg = (
            f"Project tests must be run inside the '{env_name}' conda environment.\n"
            "To create and activate it, run:\n\n"
            f"  mamba env create -f environment.yml -n {env_name} || mamba env update -f environment.yml -n {env_name}\n"
            f"  conda activate {env_name}\n\n"
            "If you intentionally want to run without the conda env, set SKIP_ENV_CHECK=1 or DATASELECTOR_IGNORE_ENV_CHECK=1 to bypass this guard (not recommended)."
        )
        pytest.exit(msg)


@pytest.fixture(scope="session", autouse=True)
def cleanup_stray_processes():
    """
    Safety fixture: Ensure no stray pipeline processes are left running after tests.
    This prevents 'zombie' processes from corrupting data or blocking resources.
    """
    yield
    # Teardown: Kill known pipeline scripts if they are still running
    # We use check=False to ignore errors if no processes are found
    subprocess.run(["pkill", "-f", "run_fine_sweep.py"], check=False)
    subprocess.run(["pkill", "-f", "run_adaptive_pipeline.py"], check=False)
    subprocess.run(["pkill", "-f", "xxl_KDR146_run_thesis_complete.py"], check=False)


@pytest.fixture
def mock_features_path(tmp_path):
    """Create a dummy features.npy file to avoid expensive model inference in tests.

    The generated feature matrix will match the number of candidates in
    `data/new_all_tiles.csv` when available to avoid shape mismatches with
    metadata used in tests.
    """
    # Prefer the real metadata size if available
    try:
        import pandas as pd

        meta_path = os.path.join(ROOT, "data", "new_all_tiles.csv")
        if os.path.exists(meta_path):
            n_samples = len(pd.read_csv(meta_path))
        else:
            n_samples = 673
    except Exception:
        n_samples = 673

    dim = 256
    features = np.random.RandomState(42).randn(n_samples, dim).astype("float32")
    path = tmp_path / "features.npy"
    np.save(path, features)
    return tmp_path
