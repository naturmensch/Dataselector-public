import pytest
import sys
import os
from pathlib import Path
import subprocess
import pytest
optuna = pytest.importorskip("optuna")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.smoke
def test_optuna_storage_creation(tmp_path):
    """Test that optuna_optimize.py creates a SQLite DB by default."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    
    # Create a dummy output dir structure
    cwd = tmp_path
    (cwd / "outputs" / "runs").mkdir(parents=True)
    
    # Run optuna_optimize.py with small trials
    cmd = [
        sys.executable,
        "-m",
        "scripts.optuna_optimize",
        "--n-trials", "2",
        "--n-candidates", "10",
        "--dim", "4",
        "--n-samples", "2",
        "--exp-name", "test_storage"
    ]
    
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Run failed: {res.stderr}"
    
    # Check if DB exists
    runs = list((cwd / "outputs" / "runs").glob("*test_storage*"))
    assert len(runs) == 1
    run_dir = runs[0]
    
    db_path = run_dir / "optuna_study.db"
    assert db_path.exists(), "optuna_study.db was not created"
    
    # Check if we can load the study
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name="kdr100_opt", storage=storage)
    assert len(study.trials) == 2

@pytest.mark.smoke
def test_import_script(tmp_path):
    """Test importing trials from CSV into Optuna storage."""
    # Create a dummy CSV
    csv_path = tmp_path / "trials.csv"
    df = pd.DataFrame({
        "number": [0, 1],
        "value": [0.5, 0.6],
        "state": ["COMPLETE", "COMPLETE"],
        "a": [0.1, 0.2],
        "b": [0.2, 0.3],
        "c": [0.7, 0.5],
        "min_distance_km": [10, 20],
        "n_samples": [5, 5]
    })
    df.to_csv(csv_path, index=False)
    
    db_path = tmp_path / "study.db"
    storage = f"sqlite:///{db_path}"
    
    cmd = [
        sys.executable,
        "-m",
        "scripts.import_trials_csv_to_optuna",
        "--csv", str(csv_path),
        "--storage", storage,
        "--study-name", "test_import"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Import failed: {res.stderr}"
    
    # Verify import
    study = optuna.load_study(study_name="test_import", storage=storage)
    assert len(study.trials) == 2
    assert study.best_value == 0.6
    assert study.best_trial.params['a'] == 0.2