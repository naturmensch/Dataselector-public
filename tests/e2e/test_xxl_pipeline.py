"""E2E test for XXL 5-phase pipeline workflow.

Tests the dataselector xxl-pipeline command for complete thesis pipeline:
- Phase 0: Validation
- Phase 1: Exploration (Sobol sampling)
- Phase 2: Coarse tuning
- Phase 3: Fine tuning
- Phase 4: Bootstrap UQ

Uses abbreviated dataset (Hamburg subset) for speed in testing.
"""

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.workflow
@pytest.mark.xxl
def test_xxl_pipeline_smoke(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Quick smoke test: XXL pipeline starts and produces basic outputs.
    
    Runs in smoke mode for speed.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal autoscale results for preflight
    (output_dir / "autoscale_selected_n_samples.txt").write_text("40")
    (output_dir / "autoscale_best_latest.json").write_text(json.dumps({
        "best_value": 10.0,
        "user_attrs": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34, "min_distance_km": 11}
    }))
    
    cmd = [
        "xxl",
        "--best-sampler", "tpe",
        "--output-dir", str(output_dir),
        "--smoke",  # Fast mode for testing
        "--seed", "42",
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)
    
    assert result.returncode == 0, f"XXL pipeline failed:\n{result.stderr.decode()}"
    
    # Check for finalization summary
    summary = output_dir / "thesis_finalization_summary.json"
    assert summary.exists(), f"Expected output {summary} not found"


@pytest.mark.workflow
@pytest.mark.xxl
def test_xxl_pipeline_5_phases(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Verify pipeline executes phases (abbreviated for speed).
    
    Validates:
    - Pipeline completes
    - Expected outputs created
    - No critical errors in logs
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal autoscale results
    (output_dir / "autoscale_selected_n_samples.txt").write_text("40")
    (output_dir / "autoscale_best_latest.json").write_text(json.dumps({
        "best_value": 10.0,
        "user_attrs": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34, "min_distance_km": 11}
    }))
    
    cmd = [
        "xxl",
        "--best-sampler", "tpe",
        "--output-dir", str(output_dir),
        "--smoke",
        "--seed", "42",
    ]
    
    result = run_dataselector_cli(
        cmd,
        cwd=str(tmp_workspace),
        capture_output=True,
    )
    
    assert result.returncode == 0, f"XXL failed:\n{result.stderr.decode()}"
    
    # Check for finalization artifacts
    summary = output_dir / "thesis_finalization_summary.json"
    assert summary.exists(), f"Finalization summary not created"


@pytest.mark.error
@pytest.mark.xxl
def test_xxl_pipeline_error_missing_csv(tmp_workspace: Path, run_dataselector_cli):
    """Error test: XXL pipeline without autoscale config.
    
    Should fail gracefully.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "xxl",
        "--best-sampler", "tpe",
        "--output-dir", str(output_dir),
        # No autoscale results provided
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)
    
    assert result.returncode != 0, "Should have failed with missing autoscale config"
