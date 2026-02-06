import json
from pathlib import Path

from scripts import xxl_KDR146_run_thesis_complete_modern as mod


def test_bootstrap_fails_without_best_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "thesis_xxl_test"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    autos = {
        "n_samples": 40,
        "alpha": 0.33,
        "beta": 0.33,
        "gamma": 0.34,
        "min_distance_km": 50,
    }

    ok = mod.phase_5_bootstrap(autos, run_dir=run_dir, dry_run=False, smoke=False)
    assert ok is False


def test_bootstrap_allows_smoke_without_best_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "thesis_xxl_test"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    autos = {
        "n_samples": 40,
        "alpha": 0.33,
        "beta": 0.33,
        "gamma": 0.34,
        "min_distance_km": 50,
    }

    ok = mod.phase_5_bootstrap(autos, run_dir=run_dir, dry_run=False, smoke=True)
    assert ok is True
