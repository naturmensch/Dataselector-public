import dataselector.workflows.xxl as mod


def test_bootstrap_fails_without_best_trial(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "runs" / "thesis_xxl_test"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "run_workflow", lambda *args, **kwargs: 1)

    ok = mod.phase_5_bootstrap(run_dir=run_dir, smoke=False)
    assert ok is False


def test_bootstrap_allows_smoke_without_best_trial(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "runs" / "thesis_xxl_test"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "run_workflow", lambda *args, **kwargs: 1)

    ok = mod.phase_5_bootstrap(run_dir=run_dir, smoke=True)
    assert ok is True


def test_bootstrap_uses_registered_cli_command(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "runs" / "thesis_xxl_test"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    call = {}

    def fake_run_workflow(workflow_name, args, smoke=False):
        call["workflow_name"] = workflow_name
        call["args"] = list(args)
        call["smoke"] = smoke
        return 0

    monkeypatch.setattr(mod, "run_workflow", fake_run_workflow)

    ok = mod.phase_5_bootstrap(run_dir=run_dir, smoke=False, seed=123)
    assert ok is True
    assert call["workflow_name"] == "bootstrap-final"
    assert call["args"] == [
        "--run-dir",
        str(run_dir),
        "--n-boot",
        "500",
        "--seed",
        "123",
    ]
