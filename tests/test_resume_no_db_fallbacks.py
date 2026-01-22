import pytest
from pathlib import Path

import pandas as pd
from tests._helpers.load_script import load_script

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def monitor():
    pytest.importorskip("optuna")
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor_test")


def _make_run_with_trials(
    tmp_path: Path, name: str = "20260119_T000000_hamburg_xxl_final", n_trials: int = 3
):
    run_dir = tmp_path / "outputs" / "runs" / name
    res_dir = run_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Write config with n_trials
    cfg = {"n_trials": n_trials}
    import yaml

    (cfg_dir / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    # Write trials.csv with n_trials COMPLETE trials
    mock = {
        "trial_number": list(range(n_trials)),
        "state": ["TrialState.COMPLETE"] * n_trials,
        "value": [50 + i for i in range(n_trials)],
        "a": [0.1] * n_trials,
        "b": [0.1] * n_trials,
        "c": [0.1] * n_trials,
        "min_distance_km": [40.0] * n_trials,
        "n_samples": [25] * n_trials,
    }
    df = pd.DataFrame(mock)
    (res_dir / "trials.csv").write_text(df.to_csv(index=False))
    return run_dir


def test_resume_uses_existing_trials_csv(monkeypatch, tmp_path, monitor):
    # Prepare run with trials.csv and config
    _run_dir = _make_run_with_trials(tmp_path, n_trials=3)

    # Monkeypatch runs_root discovery by pointing ROOT
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Monkeypatch run_hook so phases succeed
    def fake_run_hook(**kwargs):
        return {"success": True}

    monkeypatch.setattr(monitor, "run_hook", lambda **kw: {"success": True})

    # Call resume
    active_log = tmp_path / "monitor.log"
    res = monitor._resume_run("last", active_log, force=True, dry_run=False)

    assert res.get("ok") is True
    assert res.get("completed_before") == 3
    assert res.get("remaining_requested") == 0
    assert any(
        p["name"] in ("reproducibility", "finalize", "optuna")
        for p in res.get("phases", [])
    )


def test_resume_attempts_reconstruction_then_succeeds(monkeypatch, tmp_path, monitor):
    # Prepare run dir without DB or trials.csv
    run_dir = tmp_path / "outputs" / "runs" / "20260119_T000000_hamburg_xxl_final"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    import yaml

    (run_dir / "config" / "config_optuna.yaml").write_text(
        yaml.safe_dump({"n_trials": 2})
    )

    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Create a real optuna DB with completed trials so reconcile will attempt reconstruction
    import optuna

    db = run_dir / "optuna_study.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="kdr100_opt",
        load_if_exists=True,
    )

    def simple_objective(t):
        return 1.0

    study.optimize(simple_objective, n_trials=2)

    # Monkeypatch reconstruction to create a trials.csv and return True
    def fake_reconstruct(rundir, active_log, study_name=None):
        res_dir = Path(rundir) / "results"
        df = pd.DataFrame(
            {
                "trial_number": [0, 1],
                "state": ["TrialState.COMPLETE", "TrialState.COMPLETE"],
                "value": [10.0, 20.0],
                "a": [0.1, 0.2],
                "b": [0.1, 0.2],
                "c": [0.1, 0.2],
                "min_distance_km": [40.0, 40.0],
                "n_samples": [25, 25],
            }
        )
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "trials.csv").write_text(df.to_csv(index=False))
        return True

    monkeypatch.setattr(monitor, "_reconstruct_trials_from_db", fake_reconstruct)
    monkeypatch.setattr(monitor, "run_hook", lambda **kw: {"success": True})

    active_log = tmp_path / "monitor.log"
    res = monitor._resume_run("last", active_log, force=True, dry_run=False)

    assert res.get("ok") is True
    assert res.get("completed_before") == 2
    assert res.get("remaining_requested") == 0
