"""Tests for final selection workflow."""

import pytest


def test_final_selection_importable():
    """Test that final_selection module can be imported."""
    from dataselector.workflows import final_selection

    assert hasattr(final_selection, "run_final_selection")


def test_final_selection_signature():
    """Test run_final_selection function signature."""
    from dataselector.workflows.final_selection import run_final_selection
    import inspect

    sig = inspect.signature(run_final_selection)
    params = list(sig.parameters.keys())

    # Check expected parameters
    expected_params = [
        "n_samples",
        "alpha",
        "beta",
        "gamma",
        "min_distance_km",
        "use_bootstrap_best",
        "seed",
        "output_dir",
    ]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


@pytest.mark.skipif(
    True, reason="Requires full pipeline setup (features, metadata, config)"
)
def test_run_final_selection_integration():
    """Integration test for run_final_selection (skipped in CI)."""
    from dataselector.workflows.final_selection import run_final_selection
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "final_selection"

        # Would require:
        # - Valid config file
        # - Extracted features
        # - Metadata CSV
        # - Cluster labels
        sel_df, metrics = run_final_selection(
            n_samples=10,
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            min_distance_km=50.0,
            seed=42,
            output_dir=output_dir,
        )

        assert len(sel_df) == 10
        assert "alpha" in metrics
        assert metrics["n_selected"] == 10


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "final-selection", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "final-selection" in result.stdout.lower() or "usage" in result.stdout.lower()
