import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def skip_if_no_optuna():
    pytest.importorskip("optuna")

from tests._helpers.load_script import load_script


@pytest.fixture(scope="module")
def monitor_mod():
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor")


@pytest.fixture
def _resume_run(monitor_mod):
    return monitor_mod._resume_run


def test_resume_dry_run(tmp_path, monkeypatch, _resume_run):
    # create run dir under project's outputs/runs so monitor can find it by name
    run_dir_name = "pytest_test_resume_" + tmp_path.name
    run_dir = Path("outputs") / "runs" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    cfg = {"n_trials": 10}
    import yaml

    (run_dir / "config" / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    # create a small optuna study with 3 completed trials
    db_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="kdr100_opt",
        load_if_exists=True,
    )

    def simple_objective(t):
        return 1.0

    study.optimize(simple_objective, n_trials=3)

    # Create manifest.json with metadata
    manifest = {"status": "attached", "metadata": {"n_trials": 10}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    active_log = tmp_path / "monitor.log"
    active_log.write_text("")

    # Use explicit run dir name to target our tmp run
    res = _resume_run(run_dir.name, active_log, force=True, dry_run=True)
    assert res["ok"] is True
    assert res["dry_run"] is True
    # remaining should match configured n_trials minus completed (capped at 0)
    assert res["remaining"] == max(0, cfg["n_trials"] - res["completed"])


def test_resume_executes(monkeypatch, tmp_path, _resume_run):
    # create run dir under project's outputs/runs so monitor can find it by name
    run_dir_name = "pytest_test_resume_exec_" + tmp_path.name
    run_dir = Path("outputs") / "runs" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    cfg = {"n_trials": 8}
    import yaml

    (run_dir / "config" / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    db_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="kdr100_opt",
        load_if_exists=True,
    )

    def simple_objective(t):
        return 1.0

    study.optimize(simple_objective, n_trials=2)

    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "attached", "metadata": {"n_trials": 8}})
    )

    called = {}

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
        called["cmd"] = cmd_str
        return {"success": True}

    monkeypatch.setattr("scripts.xxl_full_run_monitor.run_hook", fake_run_hook)

    active_log = tmp_path / "monitor2.log"
    active_log.write_text("")

    # Use explicit run dir name to target our tmp run
    res = _resume_run(run_dir.name, active_log, force=True, dry_run=False)
    assert res["ok"] is True
    # remaining_requested should match n_trials - completed_before (capped at 0)
    assert res["remaining_requested"] == max(
        0, cfg["n_trials"] - res["completed_before"]
    )
    # If trials remain, ensure resume command targeted optuna_optimize; otherwise ensure no remaining
    if res["remaining_requested"] > 0:
        # Accept either module invocation (-m scripts.optuna_optimize) or direct script path
        phases_cmds = [p.get("cmd", "") for p in res.get("phases", [])]
        joined = " ".join(phases_cmds) + " " + str(res.get("resume_cmd", ""))
        assert "optuna_optimize" in joined
    else:
        assert res["remaining_requested"] == 0
    # ensure meta file written
    meta_file = run_dir / "results" / "resume_meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["completed_before"] == res["completed_before"]
    assert meta["completed_after"] is not None or meta["ok"]


def test_staged_resume_after_optuna_complete(monkeypatch, tmp_path, _resume_run):
    # Create run dir with optuna already completed (no remaining trials)
    run_dir_name = "pytest_test_resume_staged_" + tmp_path.name
    # Use isolated ROOT for this test to avoid picking up real outputs/artifacts
    local_root = tmp_path
    monkeypatch.setattr("scripts.xxl_full_run_monitor.ROOT", local_root)
    run_dir = local_root / "outputs" / "runs" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    cfg = {"n_trials": 4}
    import yaml

    (run_dir / "config" / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    # Create optuna DB with 4 completed trials
    db_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="kdr100_opt",
        load_if_exists=True,
    )

    def simple_objective(t):
        return 1.0

    study.optimize(simple_objective, n_trials=4)

    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "attached", "metadata": {"n_trials": 4}})
    )

    called = []

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
        called.append((name, cmd_str))
        return {"success": True}

    monkeypatch.setattr("scripts.xxl_full_run_monitor.run_hook", fake_run_hook)

    active_log = tmp_path / "monitor3.log"
    active_log.write_text("")

    res = _resume_run(run_dir.name, active_log, force=True, dry_run=False)
    assert res["ok"] is True
    # Expect phases to include reproducibility and finalize
    phase_names = [p["name"] for p in res.get("phases", [])]
    assert "reproducibility" in phase_names
    assert "finalize" in phase_names
    # Ensure our fake_run_hook was called for those phases
    called_names = [c[0] for c in called]
    assert any("reproducibility" in n for n in called_names)
    assert any("finalize" in n for n in called_names)


def test_dry_run_reports_phases_when_missing(tmp_path, monkeypatch):
    # Setup isolated root with no final selection and no repro runs
    local_root = tmp_path
    monkeypatch.setattr("scripts.xxl_full_run_monitor.ROOT", local_root)

    run_dir_name = "pytest_dry_phases_" + tmp_path.name
    run_dir = local_root / "outputs" / "runs" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    cfg = {"n_trials": 4}
    import yaml

    (run_dir / "config" / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    # Create optuna DB with 4 completed trials
    db_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="kdr100_opt",
        load_if_exists=True,
    )

    def simple_objective(t):
        return 1.0

    study.optimize(simple_objective, n_trials=4)

    # Ensure no final selection exists in local_root
    assert not (local_root / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").exists()

    active_log = tmp_path / "monitor4.log"
    active_log.write_text("")

    res = _resume_run(run_dir.name, active_log, force=True, dry_run=True)
    assert res["ok"] is True
    assert res["dry_run"] is True
    phase_names = [p["name"] for p in res.get("phases", [])]
    assert "reproducibility" in phase_names
    assert "finalize" in phase_names
