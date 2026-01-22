import json

import scripts.xxl_full_run_monitor as monitor


class BrokenExecutor:
    def __init__(self, **kw):
        pass

    def execute(self, tasks, rundir):
        raise RuntimeError("simulated executor failure")


def test_resume_handles_internal_exception(monkeypatch, tmp_path):
    """If TaskExecutor.execute raises, _resume_run should return structured error meta."""
    # Prepare a minimal run_dir with trials.csv and config
    run_dir = tmp_path / "outputs" / "runs" / "20260120_T000002_hamburg_xxl_final"
    results = run_dir / "results"
    results.mkdir(parents=True)
    # CSV with 2 completed trials
    (results / "trials.csv").write_text(
        "trial_number,state,value\n0,TrialState.COMPLETE,10\n1,TrialState.COMPLETE,20\n"
    )
    (run_dir / "config").mkdir()
    (run_dir / "config" / "config_optuna.yaml").write_text("n_trials: 2")

    # Ensure reconcile reports CSV as authoritative and target reached (remaining=0)
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

    # Replace TaskExecutor in scripts.recovery to our BrokenExecutor
    import scripts.recovery as recovery_mod

    monkeypatch.setattr(recovery_mod, "TaskExecutor", BrokenExecutor)

    # Use a noop run_hook to avoid side effects
    monkeypatch.setattr(monitor, "run_hook", lambda *a, **kw: {"success": True})
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Call resume and assert structured error meta returned
    res = monitor._resume_run("last", tmp_path / "mon.log", force=True, dry_run=False)
    assert isinstance(res, dict)
    assert res.get("ok") is False
    assert res.get("reason") == "internal_exception"
    assert "simulated executor failure" in res.get("message", "")

    # And ensure resume_meta.json is written to disk with the error info
    meta_file = run_dir / "results" / "resume_meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta.get("ok") is False
    assert meta.get("reason") == "internal_exception"
    # Expect a stacktrace and provenance info to help debugging
    assert "traceback" in meta and "simulated executor failure" in meta["traceback"]
    assert "git_sha" in meta
    assert "env" in meta and "PYTHONPATH" in meta["env"] and "PATH" in meta["env"]
