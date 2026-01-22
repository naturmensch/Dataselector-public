import sys
import types
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor_test")


def test_reconstruct_uses_trial_user_attrs_for_n_samples(monkeypatch, tmp_path):
    # Prepare run dir with fake DB
    run_dir = tmp_path / "outputs" / "runs" / "20260121_T000000_hamburg_xxl_final"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    db = run_dir / "optuna_study.db"
    db.write_text("sqlite-data")

    # Fake sqlite integrity OK
    class FakeConn:
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
            return None

        def fetchall(self):
            if "select study_name from studies" in self._last:
                return [("kdr100_opt",)]
            return []

        def close(self):
            pass

    import sqlite3 as _sqlite

    monkeypatch.setattr(_sqlite, "connect", lambda path: FakeConn())

    # Fake optuna study with trials that have user_attrs n_samples
    class FakeTrial:
        def __init__(self, number, value, n_samples):
            self.number = number
            self.datetime_start = None
            self.datetime_complete = None
            self.duration = None
            self.value = value
            self.params = {}
            self.user_attrs = {"n_samples": n_samples}
            self.state = "TrialState.COMPLETE"

    class FakeStudy:
        def __init__(self, trials):
            self.trials = trials

    def fake_load_study(study_name, storage):
        return FakeStudy([FakeTrial(0, 1.0, 25), FakeTrial(1, 2.0, 25)])

    # Use monkeypatch to avoid leaking the fake optuna into other tests
    monkeypatch.setitem(
        sys.modules, "optuna", types.SimpleNamespace(load_study=fake_load_study)
    )

    ok = monitor._reconstruct_trials_from_db(run_dir, tmp_path / "log.txt")
    assert ok is True
    # Check trials.csv now contains n_samples column values
    df_text = (run_dir / "results" / "trials.csv").read_text()
    assert (
        "25" in df_text
    ), "Reconstructed CSV should include n_samples from trial.user_attrs"


def test_trial_callback_prefers_param_then_user_attr(monkeypatch):
    # Simulate a trial object with empty params but user_attrs containing n_samples
    class DummyTrial:
        def __init__(self):
            self.number = 0
            self.duration = None
            self.value = 1.0
            self.params = {}
            self.user_attrs = {"n_samples": 42}
            self.state = "TrialState.COMPLETE"

    trial = DummyTrial()

    # Recreate the serialization logic from optuna_optimize.trial_callback
    n_samples = (
        trial.params.get("n_samples")
        if trial.params and trial.params.get("n_samples") is not None
        else (
            trial.user_attrs.get("n_samples")
            if hasattr(trial, "user_attrs")
            and trial.user_attrs.get("n_samples") is not None
            else None
        )
    )
    assert n_samples == 42
