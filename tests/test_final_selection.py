"""Tests for final selection workflow."""


def test_final_selection_importable():
    """Test that final_selection module can be imported."""
    from dataselector.workflows import final_selection

    assert hasattr(final_selection, "run_final_selection")


def test_final_selection_signature():
    """Test run_final_selection function signature."""
    import inspect

    from dataselector.workflows.final_selection import run_final_selection

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
    assert (
        "final-selection" in result.stdout.lower() or "usage" in result.stdout.lower()
    )
