"""
Tests for dataselector/workflows/optuna_optimize.py

Validates:
- Module imports cleanly
- get_optuna_sampler factory
- load_or_create_data with synthetic fallback
- CLI integration
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_optuna_optimize_importable():
    """Module should import without heavy dependencies at import-time."""
    from dataselector.workflows import optuna_optimize

    assert hasattr(optuna_optimize, "run_optuna")
    assert hasattr(optuna_optimize, "get_optuna_sampler")
    assert hasattr(optuna_optimize, "objective_factory")
    assert hasattr(optuna_optimize, "load_or_create_data")
    assert hasattr(optuna_optimize, "main")


def test_get_optuna_sampler():
    """Test sampler factory for all supported types."""
    pytest = sys.modules.get("pytest")
    if pytest:
        pytest.importorskip("optuna")
    else:
        from importlib.util import find_spec

        if find_spec("optuna") is None:
            return  # Skip test if optuna not available

    from dataselector.workflows.optuna_optimize import get_optuna_sampler

    # TPE (default)
    sampler_tpe = get_optuna_sampler("tpe", seed=42)
    assert sampler_tpe is not None
    assert "TPE" in str(type(sampler_tpe))

    # CMA-ES
    sampler_cmaes = get_optuna_sampler("cmaes", seed=42)
    assert sampler_cmaes is not None
    assert "CmaEs" in str(type(sampler_cmaes))

    # QMC (fallback to TPE if QMC not available, graceful handling)
    sampler_qmc = get_optuna_sampler("qmc-sobol", seed=42)
    assert sampler_qmc is not None


@pytest.mark.synthetic_data
def test_load_or_create_data_synthetic(tmp_path):
    """Test synthetic data generation when features/metadata don't exist."""
    from dataselector.workflows.optuna_optimize import load_or_create_data

    features, metadata = load_or_create_data(out_dir=tmp_path, n=100, dim=64, seed=123)

    assert features.shape == (100, 64)
    assert len(metadata) == 100
    assert "ul_x" in metadata.columns
    assert "ul_y" in metadata.columns
    assert "lr_x" in metadata.columns
    assert "lr_y" in metadata.columns
    assert "year" in metadata.columns


def test_cli_integration():
    """Test CLI entry point with --help (should not crash)."""
    root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, "-m", "dataselector", "optuna-optimize", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    out = f"{res.stdout}\n{res.stderr}"
    assert "--pre-names" in out
    assert "--pre-indices" in out
    assert "--hamburg" in out


def test_main_deduplicates_hamburg_preselection(monkeypatch):
    """CLI main should deduplicate Hamburg shortcut + explicit preselection."""
    from dataselector.workflows import optuna_optimize as mod

    captured = {}

    def fake_run_optuna(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mod, "run_optuna", fake_run_optuna)

    rc = mod.main(
        n_trials=1,
        pre_names=["Hamburg", "Kiel", "Hamburg"],
        pre_indices=[2, 2, 1],
        hamburg=True,
    )

    assert rc == 0
    assert captured["pre_selected_names"] == ["Hamburg", "Kiel"]
    assert captured["pre_selected_indices"] == [2, 1]
