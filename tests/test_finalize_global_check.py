import json
from pathlib import Path

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(
    ROOT / "scripts" / "xxl_full_run_monitor.py",
    module_name="scripts.xxl_full_run_monitor_test",
)


def _make_run(tmp_path: Path, name: str = "run_current_hamburg_xxl"):
    run_dir = tmp_path / "outputs" / "runs" / name
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    import yaml

    # create a config with n_trials equal to completed
    (run_dir / "config" / "config_optuna.yaml").write_text(
        yaml.safe_dump({"n_trials": 2})
    )
    # create trials.csv with 2 completed
    (run_dir / "results" / "trials.csv").write_text(
        "trial_number,state,value\n0,TrialState.COMPLETE,1\n1,TrialState.COMPLETE,2\n"
    )
    return run_dir


def test_resume_schedules_finalize_if_global_final_exists_for_other_run(
    monkeypatch, tmp_path
):
    # Prepare a run
    _run_dir = _make_run(tmp_path, name="run_current_hamburg_xxl")
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Write global final selection belonging to another run
    global_sel = {"run_id": "some_other_run", "best_value": 1.0}
    (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").write_text(
        json.dumps(global_sel)
    )

    # Monkeypatch run_hook to noop
    monkeypatch.setattr(monitor, "run_hook", lambda **kw: {"success": True})

    res = monitor._resume_run(
        "last", tmp_path / "monitor.log", force=True, dry_run=False
    )
    assert res.get("ok") is True
    assert any(
        p["name"] == "finalize" for p in res.get("phases", [])
    ), "Finalize should be scheduled when global final is for another run"


def test_resume_skips_finalize_if_global_final_matches_run(monkeypatch, tmp_path):
    # Prepare a run
    _run_dir = _make_run(tmp_path, name="run_current_hamburg_xxl")
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Write global final selection belonging to this run
    global_sel = {"run_id": "run_current_hamburg_xxl", "best_value": 1.0}
    (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").write_text(
        json.dumps(global_sel)
    )

    monkeypatch.setattr(monitor, "run_hook", lambda **kw: {"success": True})

    res = monitor._resume_run(
        "last", tmp_path / "monitor.log", force=True, dry_run=False
    )
    assert res.get("ok") is True
    assert not any(
        p["name"] == "finalize" for p in res.get("phases", [])
    ), "Finalize should be skipped when global final matches this run"
