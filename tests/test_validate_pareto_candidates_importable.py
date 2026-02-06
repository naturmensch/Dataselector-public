def test_validate_pareto_candidates_importable():
    """Test that validate_pareto_candidates module is importable."""
    # Updated to new workflow module location
    from dataselector.workflows import validation

    assert hasattr(validation, "validate_pareto_candidates")
    # Check that function exists and is callable
    assert callable(validation.validate_pareto_candidates)
