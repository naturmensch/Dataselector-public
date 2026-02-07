import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def skip_if_no_optuna():
    pytest.importorskip("optuna")


@pytest.mark.smoke
def test_optuna_storage_creation(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "ul_x": [9.95, 10.05, 10.15, 10.25],
            "ul_y": [50.05, 50.15, 50.25, 50.35],
            "lr_x": [10.05, 10.15, 10.25, 10.35],
            "lr_y": [49.95, 50.05, 50.15, 50.25],
            "year": [1900, 1901, 1902, 1903],
        }
    ).to_csv(metadata_csv, index=False)

    cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "optuna-optimize",
        "--n-trials",
        "2",
        "--n-candidates",
        "10",
        "--dim",
        "4",
        "--n-samples",
        "2",
        "--metadata-path",
        str(metadata_csv),
        "--exp-name",
        "test_storage",
    ]

    res = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Run failed: {res.stderr}"

    # study_db can be optional; ensure the main results artifact exists.
    assert (tmp_path / "outputs" / "optuna_results.csv").exists()


@pytest.mark.smoke
def test_optuna_import_command(tmp_path):
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
        "test_import",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Import failed: {res.stderr}"
    assert db_path.exists()
