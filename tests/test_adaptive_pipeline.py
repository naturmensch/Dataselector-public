#!/usr/bin/env python3
"""Tests for adaptive_pipeline workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_adaptive_pipeline_importable():
    """Test that adaptive_pipeline module is importable and has expected functions."""
    from dataselector.workflows import adaptive_pipeline

    assert hasattr(adaptive_pipeline, "run_adaptive_pipeline")
    assert hasattr(adaptive_pipeline, "main")
    assert callable(adaptive_pipeline.run_adaptive_pipeline)
    assert callable(adaptive_pipeline.main)


def test_next_power_of_two():
    """Test _next_power_of_two helper function."""
    from dataselector.workflows.adaptive_pipeline import _next_power_of_two

    assert _next_power_of_two(1) == 1
    assert _next_power_of_two(2) == 2
    assert _next_power_of_two(3) == 4
    assert _next_power_of_two(15) == 16
    assert _next_power_of_two(16) == 16
    assert _next_power_of_two(17) == 32
    assert _next_power_of_two(100) == 128


def test_normalize_n_initial_strategy_aliases():
    """Legacy strategy names must map to the modern internal contract."""
    from dataselector.workflows.adaptive_pipeline import _normalize_n_initial_strategy

    assert _normalize_n_initial_strategy("modern") == "modern"
    assert _normalize_n_initial_strategy("legacy") == "legacy"
    assert _normalize_n_initial_strategy("conservative") == "legacy"
    assert _normalize_n_initial_strategy("moderate") == "modern"
    assert _normalize_n_initial_strategy("aggressive") == "modern"
    assert _normalize_n_initial_strategy("  MODERATE  ") == "modern"


def test_normalize_n_initial_strategy_rejects_unknown_value():
    from dataselector.workflows.adaptive_pipeline import _normalize_n_initial_strategy

    with pytest.raises(ValueError, match="Unknown n_initial_strategy"):
        _normalize_n_initial_strategy("invalid")


def test_adaptive_pipeline_fails_fast_when_metadata_missing(tmp_path):
    """Missing metadata must fail immediately instead of using silent fallbacks."""
    from dataselector.workflows.adaptive_pipeline import run_adaptive_pipeline

    missing_csv = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Metadata not found"):
        run_adaptive_pipeline(csv_path=missing_csv)


def test_resolve_optuna_n_samples_prefers_explicit_value():
    from dataselector.workflows.adaptive_pipeline import _resolve_optuna_n_samples

    value, source = _resolve_optuna_n_samples(
        13,
        root=ROOT,
        context="test.adaptive",
    )
    assert value == 13
    assert source == "explicit"


def test_resolve_optuna_n_samples_reads_config(tmp_path: Path):
    from dataselector.workflows.adaptive_pipeline import _resolve_optuna_n_samples

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: 24\n",
        encoding="utf-8",
    )

    value, source = _resolve_optuna_n_samples(
        None,
        root=tmp_path,
        context="test.adaptive",
    )
    assert value == 24
    assert source == "config"


def test_resolve_optuna_n_samples_fails_without_source(tmp_path: Path):
    from dataselector.workflows.adaptive_pipeline import _resolve_optuna_n_samples

    with pytest.raises(
        ValueError, match="could not resolve selection target n_samples"
    ):
        _resolve_optuna_n_samples(
            None,
            root=tmp_path,
            context="test.adaptive",
            experiment_run_dir=tmp_path / "missing_exp",
        )


@pytest.mark.skipif(
    not (ROOT / "data" / "new_all_tiles.csv").exists(),
    reason="Requires data/new_all_tiles.csv",
)
@pytest.mark.skip(reason="Requires sklearn and full dataselector dependencies")
def test_adaptive_pipeline_dry_run(tmp_path, monkeypatch):
    """Test adaptive pipeline in dry-run mode (skip actual execution)."""
    from dataselector.workflows.adaptive_pipeline import run_adaptive_pipeline

    # Mock subprocess calls to prevent actual execution
    called_commands = []

    def mock_call(cmd, shell=True):
        called_commands.append(cmd)
        return 0

    import subprocess

    monkeypatch.setattr(subprocess, "call", mock_call)

    # Create minimal test environment
    outputs_dir = tmp_path / "outputs" / "experiments"
    outputs_dir.mkdir(parents=True)
    monkeypatch.setenv("EXPERIMENT_RUN_DIR", str(outputs_dir))

    # Create fake pareto files to satisfy existence checks
    tuning_dir = outputs_dir / "tuning_weights" / "pareto"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "pareto_solutions.csv").write_text("metric,value\n1,2\n")

    fine_dir = outputs_dir / "fine_sweep"
    fine_dir.mkdir(parents=True)
    (fine_dir / "pareto_solutions.csv").write_text("metric,value\n1,2\n")

    # Run with all phases skipped (to test orchestration only)
    try:
        run_dir = run_adaptive_pipeline(
            experiment_name="test_adaptive",
            n_lhs=8,
            n_trials=5,
            n_boot=10,
            n_candidates=100,
            sampler="sobol",
            seed=42,
            skip_exploration=True,
            skip_fine=True,
            skip_optuna=True,
            skip_bootstrap_injection=True,
            dry_run=True,
        )
        assert run_dir.exists()
        assert "test_adaptive" in str(run_dir)
    except SystemExit:
        # Expected if bootstrap command fails in mocked environment
        pass


def test_cli_integration(monkeypatch):
    """Test CLI entry point for adaptive-pipeline."""
    from dataselector.cli import main

    # Mock the workflow function to avoid actual execution
    run_called = []

    def mock_main(argv=None):
        run_called.append(argv)
        return 0

    monkeypatch.setattr("dataselector.workflows.adaptive_pipeline.main", mock_main)

    # Test help text generation (should not raise)
    with pytest.raises(SystemExit) as exc_info:
        main(["adaptive-pipeline", "--help"])
    # Help exits with 0
    assert exc_info.value.code == 0
