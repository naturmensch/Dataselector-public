"""Tests for dataselector/workflows/optuna_autoscale.py"""


def test_optuna_autoscale_importable():
    """Module should import without heavy dependencies at import-time."""
    from dataselector.workflows import optuna_autoscale

    assert hasattr(optuna_autoscale, "run_autoscale")
    assert hasattr(optuna_autoscale, "make_objective")
    assert hasattr(optuna_autoscale, "load_or_create_data")
    assert hasattr(optuna_autoscale, "clamp")
    assert hasattr(optuna_autoscale, "main")


def test_clamp():
    """Test clamp utility function."""
    from dataselector.workflows.optuna_autoscale import clamp

    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert clamp(0.5, 0.01, 1.0) == 0.5


def test_load_or_create_data_synthetic(tmp_path):
    """Test synthetic data generation when features/metadata don't exist."""
    from dataselector.workflows.optuna_autoscale import load_or_create_data

    features, metadata = load_or_create_data(out_dir=tmp_path, n=100, dim=64, seed=123)

    assert features.shape == (100, 64)
    assert len(metadata) == 100
    assert "ul_x" in metadata.columns
    assert "ul_y" in metadata.columns
    assert "lr_x" in metadata.columns
    assert "lr_y" in metadata.columns
    assert "year" in metadata.columns


def test_cli_integration():
    """Test CLI entry point with --help."""
    from dataselector.workflows.optuna_autoscale import main

    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
