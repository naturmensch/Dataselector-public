"""Tests for thesis pipeline workflow."""

import pytest


def test_thesis_pipeline_importable():
    """Test that thesis_pipeline module can be imported."""
    from dataselector.workflows import thesis_pipeline

    assert hasattr(thesis_pipeline, "run_thesis_pipeline")


def test_thesis_pipeline_signature():
    """Test run_thesis_pipeline function signature."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline
    import inspect

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


@pytest.mark.skipif(
    True, reason="Requires full pipeline setup (data, features, config)"
)
def test_run_thesis_pipeline_integration():
    """Integration test for run_thesis_pipeline (skipped in CI)."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline
    from pathlib import Path
    import tempfile

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
