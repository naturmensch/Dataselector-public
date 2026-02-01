import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(
    ROOT / "scripts" / "xxl_full_run_monitor.py",
    module_name="scripts.xxl_full_run_monitor_test",
)


class FakeStudy:
    def __init__(self, n_completed, best_value=1.0):
        class T:
            def __init__(self):
                self.state = "COMPLETE"

        self.trials = [T() for _ in range(n_completed)]
        self.best_value = best_value


def test_monitor_auto_resume_reconstructs_db_and_finalizes(monkeypatch, tmp_path):
    """E2E: If DB has more trials than CSV, monitor should reconstruct, then finalize."""
    # Prepare run dir with small trials.csv and config requesting 5 trials
    run_dir = tmp_path / "outputs" / "runs" / "20260120_T130000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Create a CSV with only 2 trials
    mock_data = {
        "trial_number": [0, 1],
        "state": ["TrialState.COMPLETE"] * 2,
        "value": [10.0, 20.0],
        "a": [0.1, 0.2],
        "b": [0.1, 0.2],
        "c": [0.1, 0.2],
        "min_distance_km": [40.0, 40.0],
        "n_samples": [25, 25],
    }
    df = pd.DataFrame(mock_data)
    (results_dir / "trials.csv").write_text(df.to_csv(index=False))

    # config expects 5 trials (so remaining will be 3 if completed becomes 5)
    import yaml

    (cfg_dir / "config_optuna.yaml").write_text(yaml.safe_dump({"n_trials": 5}))

    # create a fake DB file
    db = run_dir / "optuna_study.db"
    db.write_text("sqlite-data")

    # Monkeypatch sqlite3.connect to report OK
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

    # Provide fake optuna module in sys.modules
    def fake_load_study(study_name, storage):
        return FakeStudy(n_completed=5, best_value=99.0)

    fake_trial = types.SimpleNamespace(
        TrialState=types.SimpleNamespace(COMPLETE="COMPLETE")
    )
    fake_optuna = types.SimpleNamespace(load_study=fake_load_study, trial=fake_trial)
    # Use monkeypatch to inject fake optuna and ensure test isolation
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setitem(sys.modules, "optuna.trial", fake_trial)

    # Monkeypatch _reconstruct_trials_from_db to create a new trials.csv with 5 trials
    def fake_reconstruct(rundir, active_log):
        res_dir = Path(rundir) / "results"
        df_text = "trial_number,state,value,a,b,c,min_distance_km,n_samples\n"
        df_text += "\n".join(
            f"{i},TrialState.COMPLETE,{100+i},0.1,0.1,0.1,40.0,25" for i in range(5)
        )
        (res_dir / "trials.csv").write_text(df_text)
        return True

    monkeypatch.setattr(monitor, "_reconstruct_trials_from_db", fake_reconstruct)

    # Monkeypatch ROOT
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Monkeypatch run_hook such that finalize executes the extractor
    def fake_run_hook(
        name,
        cmd_str,
        base_log_dir,
        active_log,
        timeout,
        retries,
        env,
        start_new_session,
        pass_dry_run,
    ):
        if name == "resume_phase_finalize":
            from scripts.xxl_KDR146_run_thesis_complete import (
                _extract_xxl_final_statistics,
            )

            rc = _extract_xxl_final_statistics(tmp_path)
            return {"success": bool(rc)}
        return {"success": True}

    monkeypatch.setattr(monitor, "run_hook", fake_run_hook)

    res = monitor._resume_run(
        "last", tmp_path / "monitor.log", force=True, dry_run=False
    )

    assert res.get("ok") is True
    assert res.get("resume_source") == "reconstructed"

    # Verify final selection JSON exists and best_trial corresponds to reconstructed best
    json_file = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    assert json_file.exists()
    with json_file.open() as f:
        j = json.load(f)
    assert j["best_value"] == pytest.approx(104.0)

    # Verify resume_meta documents reconstruction attempt
    meta_file = run_dir / "results" / "resume_meta.json"
    assert meta_file.exists()
    with meta_file.open() as f:
        meta = json.load(f)
    assert any(at["step"] == "reconstruct" for at in meta.get("resume_attempts", []))
