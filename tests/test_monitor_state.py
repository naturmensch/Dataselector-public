import sqlite3

import pandas as pd
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
monitor_state = load_script(ROOT / "scripts" / "monitor_state.py", module_name="scripts.monitor_state_test")
ExperimentStateAnalyzer = monitor_state.ExperimentStateAnalyzer


def test_csv_inspection(tmp_path):
    run_dir = tmp_path / "run"
    results = run_dir / "results"
    results.mkdir(parents=True)

    df = pd.DataFrame(
        [
            {"trial_number": 1, "value": 0.5, "state": "TrialState.COMPLETE"},
            {"trial_number": 2, "value": 0.6, "state": "TrialState.COMPLETE"},
            {"trial_number": 3, "value": 0.4, "state": "TrialState.FAIL"},
        ]
    )
    df.to_csv(results / "trials.csv", index=False)

    a = ExperimentStateAnalyzer(run_dir)
    r = a.inspect()
    assert r["csv_exists"] is True
    assert r["csv_completed"] == 2
    assert abs(r["csv_best"] - 0.6) < 1e-9


def test_db_inspection_counts(tmp_path):
    run_dir = tmp_path / "run_db"
    run_dir.mkdir(parents=True)
    db = run_dir / "optuna_study.db"

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, number INTEGER, state TEXT)"
    )
    cur.execute("CREATE TABLE trial_values (trial_id INTEGER, value REAL)")
    # Insert two complete and one incomplete
    cur.execute(
        'INSERT INTO trials (trial_id, number, state) VALUES (1, 0, "TrialState.COMPLETE")'
    )
    cur.execute(
        'INSERT INTO trials (trial_id, number, state) VALUES (2, 1, "TrialState.COMPLETE")'
    )
    cur.execute(
        'INSERT INTO trials (trial_id, number, state) VALUES (3, 2, "TrialState.RUNNING")'
    )
    cur.execute("INSERT INTO trial_values (trial_id, value) VALUES (1, 0.7)")
    cur.execute("INSERT INTO trial_values (trial_id, value) VALUES (2, 0.9)")
    conn.commit()
    conn.close()

    a = ExperimentStateAnalyzer(run_dir)
    r = a.inspect()
    assert r["db_exists"] is True
    assert r["db_integrity_ok"] is True
    assert r["db_completed"] == 2
    assert abs(r["db_best"] - 0.9) < 1e-9
