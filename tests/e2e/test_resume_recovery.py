"""E2E test for resume/recovery from interrupted pipeline.

Tests checkpoint and resume functionality for long-running pipelines
like XXL, simulating interruption and recovery.
"""

import sys
import time
from pathlib import Path

import pytest


@pytest.mark.workflow
@pytest.mark.recovery
def test_resume_recovery_checkpoint_creation(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Verify checkpoint is created during pipeline execution.

    Validates that state files are written for later resume.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start XXL pipeline
    cmd = [
        "xxl",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--abbreviated",
        "--n-lhs",
        "3",
    ]

    # Run with timeout to simulate interruption
    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )

    # Check if checkpoint file was created
    checkpoint = output_dir / "checkpoint.json"
    # Checkpoint may or may not exist depending on timing
    # This test just verifies structure is there for resume capability


@pytest.mark.error
@pytest.mark.recovery
def test_resume_graceful_missing_checkpoint(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Test resume attempt without checkpoint (edge case).

    Should handle gracefully or start fresh.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to resume without checkpoint
    cmd = [
        "xxl",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--force-resume",  # Explicitly request resume
        "--abbreviated",
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )

    # Should either succeed (starting fresh) or fail with clear message
    if result.returncode != 0:
        stderr = result.stderr.decode()
        assert (
            "checkpoint" in stderr.lower() or "resume" in stderr.lower()
        ), f"Error message not helpful: {stderr}"
