"""Tests for bootstrap final selection - now using dataselector.workflows.bootstrap module."""

import pytest


def test_bootstrap_final_selection_module_exists():
    """Test that bootstrap module exists and has required functions."""
    from dataselector.workflows import bootstrap
    
    assert hasattr(bootstrap, "run_bootstrap_final")
    assert hasattr(bootstrap, "bootstrap_selection")
    assert hasattr(bootstrap, "summarize_bootstrap")
    assert hasattr(bootstrap, "jaccard")


def test_run_bootstrap_final_callable():
    """Test that run_bootstrap_final is callable."""
    from dataselector.workflows.bootstrap import run_bootstrap_final
    
    assert callable(run_bootstrap_final)
    
    # Check function signature
    import inspect
    sig = inspect.signature(run_bootstrap_final)
    assert "run_dir" in sig.parameters
    assert "n_boot" in sig.parameters
    assert "seed" in sig.parameters
