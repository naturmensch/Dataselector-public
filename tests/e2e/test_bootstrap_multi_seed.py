"""Integration test for bootstrap with multiple seeds.

Tests bootstrap-uq command reproducibility and variance.
"""

import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.bootstrap
def test_bootstrap_multi_seed_smoke(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Quick smoke test: bootstrap-uq with multiple seeds."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bootstrap-uq",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-bootstrap",
        "3",
        "--n-seeds",
        "2",
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
    )
    assert result.returncode == 0, f"bootstrap-uq failed:\n{result.stderr.decode()}"
