"""Integration test for adaptive-auto sampling.

Tests the adaptive-auto sampling command for progressive sampling.
NOTE: adaptive-auto command doesn't exist in current CLI. Skipping.
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="adaptive-auto command not implemented")


@pytest.mark.integration
@pytest.mark.adaptive
def test_adaptive_auto_samples_smoke(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Quick smoke test: adaptive-auto runs and produces outputs."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        
        "adaptive-auto",
        "--csv", str(sample_csv),
        "--output-dir", str(output_dir),
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300)
    assert result.returncode == 0, f"adaptive-auto failed:\n{result.stderr.decode()}"


@pytest.mark.integration
@pytest.mark.adaptive
def test_adaptive_auto_with_n_samples(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Test adaptive-auto with explicit n-samples parameter."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        
        "adaptive-auto",
        "--csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--n-samples", "8",
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300)
    assert result.returncode == 0
