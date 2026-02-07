"""Tests for bootstrap pareto candidates - now using dataselector.workflows.bootstrap module."""


def test_bootstrap_pareto_module_exists():
    """Test that bootstrap module exists and has required functions."""
    from dataselector.workflows import bootstrap

    assert hasattr(bootstrap, "run_bootstrap_pareto")
    assert hasattr(bootstrap, "bootstrap_candidate")
    assert hasattr(bootstrap, "jaccard")


def test_run_bootstrap_pareto_callable():
    """Test that run_bootstrap_pareto is callable."""
    from dataselector.workflows.bootstrap import run_bootstrap_pareto

    assert callable(run_bootstrap_pareto)

    # Check function signature
    import inspect

    sig = inspect.signature(run_bootstrap_pareto)
    assert "pareto_csv" in sig.parameters
    assert "n_boot" in sig.parameters
    assert "output_csv" in sig.parameters
    assert "random_seed" in sig.parameters
