"""Tests for Optuna CSV import workflow."""

from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")


def test_optuna_import_importable():
    """Test that optuna_import module can be imported."""
    from dataselector.workflows import optuna_import

    assert hasattr(optuna_import, "import_trials_from_csv")


def test_import_signature():
    """Test import_trials_from_csv function signature."""
    import inspect

    from dataselector.workflows.optuna_import import import_trials_from_csv

    sig = inspect.signature(import_trials_from_csv)
    params = list(sig.parameters.keys())

    expected_params = ["csv_path", "storage", "study_name", "direction"]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


def test_import_trials_integration():
    """Small direct integration test for import_trials_from_csv."""
    from dataselector.workflows.optuna_import import import_trials_from_csv

    with pytest.raises(FileNotFoundError):
        import_trials_from_csv(
            csv_path=Path("/nonexistent/trials.csv"),
            storage="sqlite:////tmp/unused.db",
            study_name="test_study_missing",
        )

    # Small real import path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trials.csv"
        csv_path.write_text(
            "number,value,state,alpha,beta,gamma,min_distance_km,n_samples\n"
            "0,0.5,COMPLETE,0.1,0.2,0.7,10,5\n"
            "1,0.6,COMPLETE,0.2,0.3,0.5,20,5\n",
            encoding="utf-8",
        )
        storage = f"sqlite:///{tmpdir}/test.db"
        count = import_trials_from_csv(
            csv_path=csv_path,
            storage=storage,
            study_name="test_study",
        )
        assert count == 2
        study = optuna.load_study(study_name="test_study", storage=storage)
        assert len(study.trials) == 2


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "optuna-import", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    out = f"{result.stdout}\n{result.stderr}".lower()
    assert "optuna-import" in out or "import optuna trials" in out
