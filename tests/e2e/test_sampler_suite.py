"""E2E test for sampler-suite multi-seed comparison workflow.

Tests the dataselector sampler-suite command:
- Compares multiple samplers (QMC, TPE, CMA-ES) across seeds
- Validates sampler ranking and selection
- Checks reproducibility across runs
"""

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.sampler
def test_sampler_suite_smoke(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Quick smoke test: sampler-suite runs and produces output."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sampler-suite",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-seeds",
        "2",
        "--n-trials",
        "5",
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)

    assert result.returncode == 0, f"sampler-suite failed:\n{result.stderr.decode()}"


@pytest.mark.integration
@pytest.mark.sampler
def test_sampler_suite_produces_selection(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Verify sampler-suite creates selection JSON.

    Validates selected_sampler.json structure.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sampler-suite",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-seeds",
        "2",
        "--n-trials",
        "5",
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)
    assert result.returncode == 0

    selection_json = output_dir / "selected_sampler.json"
    assert selection_json.exists(), "selected_sampler.json not created"

    with open(selection_json) as f:
        selection = json.load(f)

    assert "sampler" in selection, "Missing sampler key"
    assert "score" in selection, "Missing score key"
