import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()


def test_experiment_state_analyzer_basic(tmp_path):
    """ExperimentStateAnalyzer should detect CSV counts and DB info."""
    from scripts.monitor_state import ExperimentStateAnalyzer

    run_dir = tmp_path / "run_test"
    (run_dir / "results").mkdir(parents=True)

    # Create a simple trials.csv with states and values
    csv = run_dir / "results" / "trials.csv"
    csv.write_text(
        "trial_number,state,value,datetime_complete\n0,COMPLETE,1.0,2026-01-01T00:00:00Z\n1,FAIL,0.5,\n2,COMPLETE,0.9,2026-01-01T01:00:00Z\n"
    )

    analyzer = ExperimentStateAnalyzer(run_dir)
    st = analyzer.inspect()

    assert st["csv_exists"] is True
    assert st["csv_completed"] == 2
    assert st["csv_best"] == pytest.approx(1.0)

    # DB absent by default
    assert st["db_exists"] is False


@pytest.mark.e2e
def test_resume_with_missing_monitor_state(tmp_path, monkeypatch):
    """If `scripts.monitor_state` is missing, the monitor should still perform a dry-run resume when RecoveryPlanner is present."""
    # Note: this test runs _resume_run in-process and does not need the data_symlink env fixture.

    runs_root = REPO_ROOT / "outputs" / "runs"
    test_run = runs_root / "incomplete_monitor_import_fallback"
    # Ensure a clean slate
    if test_run.exists():
        shutil.rmtree(test_run)
    test_run.mkdir(parents=True)
    (test_run / "config").mkdir()
    (test_run / "config" / "config_optuna.yaml").write_text("n_trials: 10\n")

    # Create an optuna DB using the same helper as other tests
    import numpy as _np
    import pandas as _pd

    from scripts.optuna_optimize import run_optuna

    features = _np.random.RandomState(1).randn(20, 16).astype("float32")
    metadata = _pd.DataFrame(
        {
            "N": _np.random.uniform(48, 55, 20),
            "left": _np.random.uniform(6, 15, 20),
            "year": _np.random.randint(1880, 1945, 20),
        }
    )

    monkeypatch.setattr(
        "scripts.optuna_optimize.load_or_create_data",
        lambda n, dim, seed: (features, metadata),
        raising=False,
    )

    class DummySelector:
        def __init__(self, n_samples, use_multi_criteria=True):
            self.n_samples = n_samples

        def select(
            self,
            features,
            metadata,
            spatial_constraint,
            min_distance_km,
            alpha_visual,
            beta_spatial,
            gamma_temporal,
        ):
            n = min(self.n_samples, len(features))
            return list(range(n))

        def _calculate_diversity_score(self, selected_features):
            return float(_np.mean(_np.var(selected_features, axis=0)))

    monkeypatch.setattr(
        "scripts.optuna_optimize.DiversitySelector", DummySelector, raising=False
    )

    run_optuna(
        n_trials=3,
        n_candidates=20,
        dim=16,
        n_samples=5,
        out_dir=test_run,
        study_db=str(test_run / "optuna_study.db"),
    )
    assert (test_run / "optuna_study.db").exists()

    # Ensure RecoveryPlanner is importable
    try:
        from scripts.recovery import RecoveryPlanner
    except Exception:
        pytest.skip("RecoveryPlanner not importable; skipping fallback test")

    # Remove scripts.monitor_state from modules to simulate import failure
    monkeypatch.delitem(sys.modules, "scripts.monitor_state", raising=False)

    # Call _resume_run directly (in-process) with dry-run flag
    from scripts.xxl_full_run_monitor import _resume_run

    active_log = tmp_path / "monitor_active.log"
    res = _resume_run(str(test_run), active_log, force=True, dry_run=True)

    assert res.get("ok") is True
    phases = [p["name"] for p in res.get("phases", [])]
    # Expect optuna phase planned (remaining trials > 0)
    assert any(p == "optuna" for p in phases)
