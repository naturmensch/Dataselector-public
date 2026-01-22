import sys
import types
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor_test")


class FakeStudy:
    def __init__(self, n_completed, best_value=1.0):
        class T:
            def __init__(self):
                self.state = "COMPLETE"

        self.trials = [T() for _ in range(n_completed)]
        self.best_value = best_value


def test_db_newer_reconstruct_and_finalize(monkeypatch, tmp_path):
    """If DB has more trials than CSV -> reconstruct then finalize."""
    run_dir = tmp_path / "outputs" / "runs" / "20260120_T000000_hamburg_xxl_final"
    results = run_dir / "results"
    results.mkdir(parents=True)

    # small CSV
    (results / "trials.csv").write_text(
        "trial_number,state,value\n0,TrialState.COMPLETE,10\n1,TrialState.COMPLETE,20\n"
    )
    # config expects 5
    (run_dir / "config").mkdir()
    (run_dir / "config" / "config_optuna.yaml").write_text("n_trials: 5")

    # fake DB candidate and sqlite integritiy ok
    db = run_dir / "optuna_study.db"
    db.write_text("sqlite-binary")

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

    # fake optuna.load_study to indicate DB has 5 completed trials
    def fake_load_study(study_name, storage):
        return FakeStudy(n_completed=5, best_value=99.0)

    fake_trial = types.SimpleNamespace(
        TrialState=types.SimpleNamespace(COMPLETE="COMPLETE")
    )
    fake_optuna = types.SimpleNamespace(load_study=fake_load_study, trial=fake_trial)
    # Use monkeypatch for safe module injection
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setitem(sys.modules, "optuna.trial", fake_trial)

    # monkeypatch reconstruct to write a new trials.csv and return True
    def fake_recon(rundir, active_log):
        rdir = Path(rundir) / "results"
        txt = "trial_number,state,value,a,b,c,min_distance_km,n_samples\n"
        txt += "\n".join(
            f"{i},TrialState.COMPLETE,{100+i},0,0,0,40,25" for i in range(5)
        )
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "trials.csv").write_text(txt)
        return True

    monkeypatch.setattr(monitor, "_reconstruct_trials_from_db", fake_recon)

    # fake run_hook such that finalize executes extractor
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
        if name == "resume_phase_finalize" or name == "finalize":
            from scripts.xxl_KDR146_run_thesis_complete import (
                _extract_xxl_final_statistics,
            )

            rc = _extract_xxl_final_statistics(tmp_path)
            return {"success": bool(rc)}
        return {"success": True}

    monkeypatch.setattr(monitor, "run_hook", fake_run_hook)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    res = monitor._resume_run("last", tmp_path / "mon.log", force=True, dry_run=False)
    assert res.get("ok") is True
    assert res.get("resume_source") == "reconstructed"
    assert (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").exists()


def test_optuna_remaining_resume_and_finalize(monkeypatch, tmp_path):
    """If CSV shows incomplete trials (remaining>0), schedule optuna resume then finalize."""
    run_dir = tmp_path / "outputs" / "runs" / "20260120_T000001_hamburg_xxl_final"
    results = run_dir / "results"
    results.mkdir(parents=True)

    # CSV has 2 completed but config expects 6
    (results / "trials.csv").write_text(
        "trial_number,state,value\n0,TrialState.COMPLETE,10\n1,TrialState.COMPLETE,20\n"
    )
    (run_dir / "config").mkdir()
    (run_dir / "config" / "config_optuna.yaml").write_text("n_trials: 6")

    # Make reconcile return that CSV is authoritative (no DB) -> completed=2
    # Monkeypatch _reconcile_trials to return this explicit result
    def fake_reconcile(rd, active_log):
        return {
            "ok": True,
            "source": "trials_csv",
            "completed_count": 2,
            "best_value": 20.0,
            "actions": [],
            "reason": None,
            "db_path": None,
            "attempts": [],
        }

    monkeypatch.setattr(monitor, "_reconcile_trials", fake_reconcile)

    # Monkeypatch run_cmd_with_retry to simulate optuna resume exit 0
    monkeypatch.setattr(monitor, "run_cmd_with_retry", lambda cmd, **kw: 0)

    # Monkeypatch reconstruct to be no-op
    monkeypatch.setattr(monitor, "_reconstruct_trials_from_db", lambda rr, al: True)

    # Monkeypatch finalize hook to run extractor
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
        if name == "resume_phase_finalize" or name == "finalize":
            from scripts.xxl_KDR146_run_thesis_complete import (
                _extract_xxl_final_statistics,
            )

            rc = _extract_xxl_final_statistics(tmp_path)
            return {"success": bool(rc)}
        return {"success": True}

    monkeypatch.setattr(monitor, "run_hook", fake_run_hook)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    res = monitor._resume_run("last", tmp_path / "mon.log", force=True, dry_run=False)
    assert res.get("ok") is True
    # ensure optuna cmd was planned/executed (check phases)
    phases = res.get("phases", [])
    assert any(p["name"] == "optuna" for p in phases) or any(
        t["name"] == "optuna" for t in res.get("phases", [])
    )
    assert (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").exists()
