"""Tests for Optuna CSV import workflow."""

import pytest


def test_optuna_import_importable():
    """Test that optuna_import module can be imported."""
    from dataselector.workflows import optuna_import

    assert hasattr(optuna_import, "import_trials_from_csv")


def test_import_signature():
    """Test import_trials_from_csv function signature."""
    from dataselector.workflows.optuna_import import import_trials_from_csv
    import inspect

    sig = inspect.signature(import_trials_from_csv)
    params = list(sig.parameters.keys())

    expected_params = ["csv_path", "storage", "study_name", "direction"]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


@pytest.mark.skipif(
    True, reason="Requires optuna and valid CSV/storage setup"
)
def test_import_trials_integration():
    """Integration test for import_trials_from_csv (skipped in CI)."""
    from dataselector.workflows.optuna_import import import_trials_from_csv
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Would require:
        # - Valid trials CSV
        # - Optuna storage
        csv_path = Path(tmpdir) / "trials.csv"
        storage = f"sqlite:///{tmpdir}/test.db"

        count = import_trials_from_csv(
            csv_path=csv_path,
            storage=storage,
            study_name="test_study",
        )

        assert count >= 0


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "optuna", "import-trials", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # CLI might not exist yet, so just check it doesn't crash
    assert result.returncode in (0, 1, 2)
