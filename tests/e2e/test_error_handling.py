"""Error handling tests for edge cases and failure modes."""

from pathlib import Path

import pytest


@pytest.mark.error
def test_cli_invalid_command(run_dataselector_cli):
    """Error test: Invalid command."""
    cmd = ["nonexistent-command"]
    result = run_dataselector_cli(cmd, capture_output=True)
    assert result.returncode != 0


@pytest.mark.error
def test_autoscale_invalid_csv(tmp_workspace: Path, run_dataselector_cli):
    """Error test: autoscale with non-CSV file."""
    invalid_csv = tmp_workspace / "data" / "invalid.txt"
    invalid_csv.parent.mkdir(parents=True, exist_ok=True)
    invalid_csv.write_text("this is not valid csv")

    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(invalid_csv),
        "--output-dir",
        str(output_dir),
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )
    # Should fail or at least produce error
    assert result.returncode != 0 or len(result.stderr) > 0


@pytest.mark.error
def test_autoscale_insufficient_samples(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Error test: autoscale with n-samples > available tiles."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-samples",
        "1000",  # More than sample_csv has
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )
    # Should handle gracefully
    if result.returncode != 0:
        stderr = result.stderr.decode()
        assert any(
            word in stderr.lower() for word in ["sample", "available", "tile", "exceed"]
        ), f"Error message not helpful: {stderr}"


@pytest.mark.error
def test_invalid_parameter_value(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Error test: Invalid parameter value."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
        "--n-trials",
        "invalid",  # Should be int
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )
    assert result.returncode != 0, "Should reject invalid parameter"


@pytest.mark.error
def test_output_dir_permission_denied(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Error test: Output dir with insufficient permissions.

    This test is platform-dependent and may not work on all systems.
    """
    # Create read-only directory
    readonly_dir = tmp_workspace / "readonly"
    readonly_dir.mkdir(parents=True, exist_ok=True)

    # Try to make read-only (may not work on all systems)
    import os

    os.chmod(readonly_dir, 0o444)

    output_dir = readonly_dir / "outputs"

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--output-dir",
        str(output_dir),
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )

    # Restore permissions for cleanup
    os.chmod(readonly_dir, 0o755)

    # May or may not fail depending on permissions
    # Just check that error message is meaningful if it fails
    if result.returncode != 0:
        stderr = result.stderr.decode()
        assert len(stderr) > 0, "Error message should be provided"
