import json
from pathlib import Path

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
