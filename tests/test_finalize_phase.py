import json
import sys
from pathlib import Path

from scripts import xxl_KDR146_run_thesis_complete_modern as mod


def test_finalize_without_run_dir_fails(tmp_path, monkeypatch):
    # Isolated project root without any runs
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    monkeypatch.setattr(sys, "argv", ["xxl", "--phase", "finalize", "--skip-env-check"])
    rc = mod.main()
    assert rc == 1


def test_finalize_with_run_dir_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    # Create a fake run dir with a best_trial.json so bootstrap/finalize can proceed
    run = tmp_path / "outputs" / "runs" / "rr1"
    (run / "results").mkdir(parents=True, exist_ok=True)
    best = {"a": 0.2, "b": 0.3, "c": 0.5, "min_distance_km": 50, "n_samples": 40}
    (run / "results" / "best_trial.json").write_text(json.dumps(best))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xxl",
            "--phase",
            "finalize",
            "--run-dir",
            str(run),
            "--dry-run",
            "--skip-env-check",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert (tmp_path / "outputs" / "thesis_finalization_summary.json").exists()
