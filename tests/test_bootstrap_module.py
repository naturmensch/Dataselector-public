"""Tests for dataselector.workflows.bootstrap module.

Tests the new direct function implementations (non-subprocess).
"""

import pandas as pd
import pytest


def test_jaccard():
    """Test jaccard similarity function."""
    from dataselector.workflows.bootstrap import jaccard

    # Identical sets
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0

    # No overlap
    assert jaccard([1, 2], [3, 4]) == 0.0

    # Partial overlap
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)

    # Empty sets
    assert jaccard([], []) == 1.0
    assert jaccard([1], []) == 0.0
    assert jaccard([], [1]) == 0.0


def test_bootstrap_selection_importable():
    """Test that bootstrap_selection function is importable."""
    from dataselector.workflows.bootstrap import bootstrap_selection

    assert callable(bootstrap_selection)


def test_bootstrap_candidate_importable():
    """Test that bootstrap_candidate function is importable."""
    from dataselector.workflows.bootstrap import bootstrap_candidate

    assert callable(bootstrap_candidate)


def test_run_bootstrap_final_importable():
    """Test that run_bootstrap_final function is importable."""
    from dataselector.workflows.bootstrap import run_bootstrap_final

    assert callable(run_bootstrap_final)


def test_run_bootstrap_pareto_importable():
    """Test that run_bootstrap_pareto function is importable."""
    from dataselector.workflows.bootstrap import run_bootstrap_pareto

    assert callable(run_bootstrap_pareto)


def test_summarize_bootstrap():
    """Test summarize_bootstrap function."""
    from dataselector.workflows.bootstrap import summarize_bootstrap

    # Create mock bootstrap results
    df_boot = pd.DataFrame(
        {
            "n_selected": [100, 102, 98, 101],
            "temporal_std": [5.0, 5.2, 4.8, 5.1],
            "spatial_mean_km": [10.0, 10.5, 9.5, 10.2],
            "jaccard_with_original": [0.9, 0.92, 0.88, 0.91],
        }
    )

    original_metrics = {
        "n_selected": 100,
        "temporal_std": 5.0,
        "spatial_mean_km": 10.0,
    }

    summary = summarize_bootstrap(df_boot, original_metrics)

    # Check that summary contains expected keys
    assert "n_selected_mean" in summary
    assert "n_selected_std" in summary
    assert "n_selected_ci_lower" in summary
    assert "n_selected_ci_upper" in summary
    assert "n_selected_original" in summary

    # Check that values are reasonable
    assert summary["n_selected_mean"] == pytest.approx(100.25, rel=0.1)
    assert summary["n_selected_std"] > 0
    assert summary["n_selected_original"] == 100


def test_bootstrap_module_no_subprocess():
    """Verify that the bootstrap module doesn't use subprocess anymore."""
    import inspect

    from dataselector.workflows import bootstrap

    # Check that run_bootstrap_final and run_bootstrap_pareto don't use subprocess
    run_bootstrap_final_source = inspect.getsource(bootstrap.run_bootstrap_final)
    run_bootstrap_pareto_source = inspect.getsource(bootstrap.run_bootstrap_pareto)

    assert (
        "subprocess.call" not in run_bootstrap_final_source
    ), "run_bootstrap_final should not use subprocess"
    assert (
        "subprocess.call" not in run_bootstrap_pareto_source
    ), "run_bootstrap_pareto should not use subprocess"
