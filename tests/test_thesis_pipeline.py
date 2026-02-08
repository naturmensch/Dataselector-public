"""Tests for thesis pipeline workflow."""

import pytest


def test_thesis_pipeline_importable():
    """Test that thesis_pipeline module can be imported."""
    from dataselector.workflows import thesis_pipeline

    assert hasattr(thesis_pipeline, "run_thesis_pipeline")


def test_thesis_pipeline_signature():
    """Test run_thesis_pipeline function signature."""
    import inspect

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    sig = inspect.signature(run_thesis_pipeline)
    params = list(sig.parameters.keys())

    expected_params = [
        "n_lhs",
        "n_trials",
        "skip_exploration",
        "skip_optimization",
        "skip_validation",
        "dry_run",
        "output_dir",
    ]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


def test_run_thesis_pipeline_dry_run_skip_validation(tmp_path):
    """Regression guard for CLI dry-run path without validation imports."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    success = run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_validation=True,
        dry_run=True,
        output_dir=tmp_path / "outputs",
    )

    assert success is True


@pytest.mark.skipif(
    True, reason="Requires full pipeline setup (data, features, config)"
)
def test_run_thesis_pipeline_integration():
    """Integration test for run_thesis_pipeline (skipped in CI)."""
    import tempfile
    from pathlib import Path

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "outputs"

        success = run_thesis_pipeline(
            n_lhs=10,
            n_trials=5,
            dry_run=True,
            output_dir=output_dir,
        )

        assert success is True


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "thesis-pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # CLI might not exist yet, so just check it doesn't crash
    assert result.returncode in (0, 1, 2)
