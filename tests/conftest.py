# Global pytest configuration and fixtures
import os
import sys
import pytest
import numpy as np
import subprocess

# Ensure repository root is on sys.path so `src` can be imported in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
        meta_path = os.path.join(ROOT, 'data', 'new_all_tiles.csv')
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
