#!/usr/bin/env python3
"""Tests for thesis_sampler_suite workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_thesis_sampler_suite_importable():
    """Test that thesis_sampler_suite module is importable and has expected functions."""
    from dataselector.workflows import thesis_sampler_suite

    assert hasattr(thesis_sampler_suite, "run_thesis_sampler_suite")
    assert hasattr(thesis_sampler_suite, "choose_best_sampler")
    assert hasattr(thesis_sampler_suite, "run_cmd")
    assert hasattr(thesis_sampler_suite, "main")
    assert callable(thesis_sampler_suite.run_thesis_sampler_suite)
    assert callable(thesis_sampler_suite.choose_best_sampler)
    assert callable(thesis_sampler_suite.main)


def test_choose_best_sampler(tmp_path):
    """Test choose_best_sampler function."""
    from dataselector.workflows.thesis_sampler_suite import choose_best_sampler

    # Create test data structure
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create fake dataset directories with summaries
    for dataset in ["hamburg", "kdr100"]:
        dataset_dir = results_dir / dataset
        dataset_dir.mkdir()
        summary = dataset_dir / "summary.csv"
        summary.write_text("sampler,mean\nqmc,0.85\ntpe,0.78\ncmaes,0.72\n")

    best, table = choose_best_sampler(results_dir)
    assert best == "qmc"  # Highest mean
    assert len(table) == 3
    assert "sampler" in table.columns
    assert "mean" in table.columns


def test_choose_best_sampler_no_summaries(tmp_path):
    """Test choose_best_sampler raises error when no summaries found."""
    from dataselector.workflows.thesis_sampler_suite import choose_best_sampler

    results_dir = tmp_path / "empty_results"
    results_dir.mkdir()

    with pytest.raises(RuntimeError, match="No summary files found"):
        choose_best_sampler(results_dir)


def test_cli_integration(monkeypatch):
    """Test CLI entry point for thesis-sampler-suite."""
    from dataselector.cli import main

    # Test help text generation (should not raise)
    with pytest.raises(SystemExit) as exc_info:
        main(["thesis-sampler-suite", "--help"])
    # Help exits with 0
    assert exc_info.value.code == 0
