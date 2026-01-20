from pathlib import Path
import json
import sqlite3
import types

import pytest

from scripts import xxl_full_run_monitor as monitor


class FakeConn:
    def __init__(self, integrity_ok=True, study_names=None):
        self._integrity_ok = integrity_ok
        self._study_names = study_names or ['kdr100_opt']
        self._last_query = None
    def cursor(self):
        return self
    def execute(self, q):
        self._last_query = q.strip().lower()
        return self
    def fetchone(self):
        if self._last_query and self._last_query.startswith('pragma integrity_check'):
            return ('ok',) if self._integrity_ok else ('fail',)
        return None
    def fetchall(self):
        if 'select study_name from studies' in (self._last_query or ''):
            return [(n,) for n in self._study_names]
        return []
    def close(self):
        pass


class FakeStudy:
    def __init__(self, n_completed, best_value=1.0):
        class T:
            def __init__(self):
                self.state = 'COMPLETE'
        self.trials = [T() for _ in range(n_completed)]
        self.best_value = best_value


def test_reconcile_db_more_trials_triggers_reconstruct(monkeypatch, tmp_path):
    run_dir = tmp_path / 'outputs' / 'runs' / '20260120_T000000_hamburg_xxl_final'
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    # write a small csv with fewer trials
    (run_dir / 'results' / 'trials.csv').write_text('trial_number,state,value\n0,TrialState.COMPLETE,10\n1,TrialState.COMPLETE,20\n')
    # create a fake db file
    db = run_dir / 'optuna_study.db'
    db.write_text('sqlite-data')

    # Monkeypatch sqlite3.connect to return fake conn with 5 completed
    def fake_connect(path):
        return FakeConn(integrity_ok=True, study_names=['kdr100_opt'])
    import sqlite3 as _sqlite
    monkeypatch.setattr(_sqlite, 'connect', fake_connect)

    # Monkeypatch optuna.load_study to return fake study with 5 completed
    def fake_load_study(study_name, storage):
        return FakeStudy(n_completed=5, best_value=99.9)
    import sys as _sys
    # Provide fake optuna module and submodule with TrialState
    fake_trial = types.SimpleNamespace(TrialState=types.SimpleNamespace(COMPLETE='COMPLETE'))
    fake_optuna = types.SimpleNamespace(load_study=fake_load_study, trial=fake_trial)
    # Use monkeypatch to set into sys.modules and ensure teardown
    monkeypatch.setitem(_sys.modules, 'optuna', fake_optuna)
    monkeypatch.setitem(_sys.modules, 'optuna.trial', fake_trial)

    # Monkeypatch reconstruct to create new trials.csv and return True
    def fake_reconstruct(rundir, active_log):
        res_dir = Path(rundir) / 'results'
        df_text = 'trial_number,state,value\n' + '\n'.join(f"{i},TrialState.COMPLETE,{100+i}" for i in range(5))
        (res_dir / 'trials.csv').write_text(df_text)
        return True
    monkeypatch.setattr(monitor, '_reconstruct_trials_from_db', fake_reconstruct)

    rec = monitor._reconcile_trials(run_dir, tmp_path / 'log.txt')
    assert rec['ok'] is True
    assert rec['source'] == 'reconstructed'
    assert rec['completed_count'] == 5
    assert 'reconstructed_from_db' in rec['actions']
    assert any(a['step'] == 'reconstruct' for a in rec['attempts'])


def test_reconcile_db_corrupt_and_no_csv_aborts(monkeypatch, tmp_path):
    run_dir = tmp_path / 'outputs' / 'runs' / '20260120_T000001_hamburg_xxl_final'
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    db = run_dir / 'optuna_study.db'
    db.write_text('bad')

    # fake sqlite connect that returns corrupted integrity
    def fake_connect(path):
        return FakeConn(integrity_ok=False)
    import sqlite3 as _sqlite
    monkeypatch.setattr(_sqlite, 'connect', fake_connect)

    rec = monitor._reconcile_trials(run_dir, tmp_path / 'log2.txt')
    assert rec['ok'] is False
    assert rec['reason'] == 'db_corrupt'
    assert any(a.startswith('db_corrupt') for a in rec['actions'])
