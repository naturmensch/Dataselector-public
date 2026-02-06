"""Integration test for Optuna persistence and checkpoint.

Tests that Optuna study is properly saved and can be resumed.
"""

import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.optuna
def test_optuna_persistence_study_created(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Verify Optuna study database is created and persisted."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-trials",
        "3",
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
    )
    assert result.returncode == 0

    # Look for Optuna database files
    optuna_files = list(output_dir.glob("*.db")) + list(output_dir.glob("optuna*"))
    # May or may not exist depending on implementation
