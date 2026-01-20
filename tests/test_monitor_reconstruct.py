import sqlite3
import tempfile
from pathlib import Path

import pytest
optuna = pytest.importorskip('optuna')
from scripts.xxl_full_run_monitor import _reconstruct_trials_from_db


def make_test_study(db_path: Path, n: int = 5):
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(direction='maximize', storage=storage_url, study_name='test_study')
    def objective(trial):
        x = trial.suggest_float('a', 0.0, 1.0)
        return x
    for _ in range(n):
        study.optimize(objective, n_trials=1)
    return study


def test_reconstruct_creates_csv(tmp_path, capsys):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    db_path = run_dir / 'optuna_study.db'
    make_test_study(db_path, n=3)

    active_log = run_dir / 'monitor.log'
    ok = _reconstruct_trials_from_db(run_dir, active_log, study_name='test_study')
    assert ok
    out = run_dir / 'results' / 'trials.csv'
    assert out.exists()
    content = out.read_text()
    assert 'trial_number' in content
    assert 'a' in content


def test_reconstruct_handles_missing_db(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    active_log = run_dir / 'monitor.log'
    ok = _reconstruct_trials_from_db(run_dir, active_log)
    assert not ok
