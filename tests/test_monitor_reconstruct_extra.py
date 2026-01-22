import json
import sys
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor_test")


class FakeConnRows:
    def __init__(self):
        self._last = ""

    def cursor(self):
        return self

    def execute(self, q):
        self._last = q.strip().lower()
        return self

    def fetchone(self):
        if self._last.startswith("pragma integrity_check"):
            return ("ok",)
        if self._last.startswith("select study_name from studies"):
            return [("kdr100_opt",)]
        return None

    def fetchall(self):
        # For trials table
        if self._last.startswith("select trial_id, number, datetime_start"):
            # return 3 trials
            return [
                (1, 0, None, None, "COMPLETE"),
                (2, 1, None, None, "COMPLETE"),
                (3, 2, None, None, "RUNNING"),
            ]
        if self._last.startswith(
            "select trial_id, param_name, param_value from trial_params"
        ):
            return [
                (1, "a", "0.1"),
                (1, "n_samples", "20"),
                (2, "a", "0.2"),
                (2, "n_samples", "20"),
            ]
        if self._last.startswith(
            "select trial_id, key, value from trial_user_attributes"
        ):
            return [(1, "n_samples", "20"), (2, "n_samples", "20")]
        if self._last.startswith("select trial_id, value from trial_values"):
            return [(1, 10.0), (2, 20.0)]
        return []

    def close(self):
        pass


def test_reconstruct_falls_back_to_direct_sqlite_when_optuna_missing(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # create candidate db file
    db = run_dir / "optuna_study.db"
    db.write_text("sqlite")

    # Remove optuna from sys.modules to simulate not installed
    if "optuna" in sys.modules:
        tmp = sys.modules.pop("optuna")
    else:
        tmp = None

    # Monkeypatch sqlite3.connect
    import sqlite3 as _sqlite

    monkeypatch.setattr(_sqlite, "connect", lambda path: FakeConnRows())

    active_log = run_dir / "monitor.log"
    ok = monitor._reconstruct_trials_from_db(run_dir, active_log)
    assert ok is True
    out = run_dir / "results" / "trials.csv"
    assert out.exists()
    txt = out.read_text()
    assert "trial_number" in txt
    # restore optuna in sys.modules if removed
    if tmp is not None:
        sys.modules["optuna"] = tmp


def test_reconstruct_picks_most_recent_candidate(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # create two candidate DBs
    older = run_dir / "optuna_study.db.bak_old"
    newer = run_dir / "optuna_study.db.bak_new"
    older.write_text("older")
    newer.write_text("newer")
    # tweak mtimes
    import os

    # set mtimes via os.utime
    os.utime(str(older), (1, 1))
    os.utime(str(newer), (2, 2))

    # monkeypatch sqlite.connect to our fake and ensure metadata writes include source_db
    import sqlite3 as _sqlite

    monkeypatch.setattr(_sqlite, "connect", lambda path: FakeConnRows())

    active_log = run_dir / "monitor.log"
    ok = monitor._reconstruct_trials_from_db(run_dir, active_log)
    assert ok is True
    meta = run_dir / "results" / "trials_reconstruct_meta.json"
    assert meta.exists()
    metaj = json.loads(meta.read_text())
    # Ensure source_db points to 'newer'
    assert "bak_new" in metaj.get("source_db")


def test_reconcile_db_corrupt_but_csv_present_uses_csv(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "run_corrupt"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "results" / "trials.csv").write_text(
        "trial_number,state,value\n0,TrialState.COMPLETE,1\n"
    )

    db = run_dir / "optuna_study.db"
    db.write_text("corrupt")

    # fake sqlite connect that returns corrupted integrity
    class BadConn:
        def __init__(self):
            self._last = ""

        def cursor(self):
            return self

        def execute(self, q):
            self._last = q.strip().lower()
            return self

        def fetchone(self):
            if self._last.startswith("pragma integrity_check"):
                return ("fail",)
            return None

        def fetchall(self):
            return []

        def close(self):
            pass

    import sqlite3 as _sqlite

    monkeypatch.setattr(_sqlite, "connect", lambda path: BadConn())

    rec = monitor._reconcile_trials(run_dir, tmp_path / "log.txt")
    assert rec["ok"] is True
    assert rec["source"] in ("csv", "trials_csv")
    assert any(a == "used_csv_due_to_db_corruption" for a in rec["actions"])
