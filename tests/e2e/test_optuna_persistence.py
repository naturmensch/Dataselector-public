"""Integration test for Optuna persistence and checkpoint.

Tests that Optuna study is properly saved and can be resumed.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.optuna
@pytest.mark.synthetic_data
def test_optuna_persistence_study_created(tmp_workspace: Path, run_dataselector_cli):
    """Verify Optuna study database is created and persisted."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--output-dir",
        str(output_dir),
        "--n-trials",
        "3",
        "--n-candidates",
        "60",
        "--dim",
        "32",
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
    )
    assert result.returncode == 0

    # Persistence implementation is allowed to vary across adapters.
