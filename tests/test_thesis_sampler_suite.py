#!/usr/bin/env python3
"""Tests for thesis_sampler_suite workflow."""

from __future__ import annotations

import json
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


def test_map_sampler_for_adaptive_pipeline_rejects_unknown():
    """Unknown sampler IDs must fail with a clear contract error."""
    from dataselector.workflows.thesis_sampler_suite import (
        map_sampler_for_adaptive_pipeline,
    )

    with pytest.raises(RuntimeError, match="Unsupported best sampler"):
        map_sampler_for_adaptive_pipeline("does-not-exist")


def test_run_suite_maps_best_sampler_for_adaptive_pipeline(monkeypatch, tmp_path):
    """Adaptive pipeline call must use valid exploration and Optuna sampler args."""
    from dataselector.workflows import thesis_sampler_suite as mod

    commands: list[tuple[list[str], dict | None]] = []

    class _FakeTable:
        def to_dict(self, orient: str = "records"):
            return []

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "run_cmd",
        lambda cmd, cwd=None, env=None: commands.append((cmd, env)),
    )
    monkeypatch.setattr(
        mod,
        "choose_best_sampler",
        lambda _results_dir: ("tpe", _FakeTable()),
    )

    mod.run_thesis_sampler_suite(
        seeds=[42],
        n_trials=2,
        datasets=["hamburg"],
        samplers=["tpe"],
        sequential=True,
        n_trials_full=4,
        n_candidates=16,
        autoscale=False,
        execution_profile="thesis_repro",
    )

    adaptive_cmds = [cmd for cmd, _env in commands if "adaptive-pipeline" in cmd]
    assert len(adaptive_cmds) == 2
    for cmd in adaptive_cmds:
        assert cmd[cmd.index("--sampler") + 1] == "lhs"
        assert cmd[cmd.index("--optuna-sampler") + 1] == "TPESampler"

    for _cmd, env in commands:
        assert env is not None
        assert env["DATASELECTOR_EXECUTION_PROFILE"] == "thesis_repro"

    suite_path = next((tmp_path / "outputs" / "runs").glob("sampler_thesis_suite_*"))
    metadata = json.loads(
        suite_path.joinpath("run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["execution_profile"] == "thesis_repro"


def test_sampler_suite_alias_forwards_arguments(monkeypatch):
    """Alias command must forward full argument contract to thesis-sampler-suite."""
    from dataselector.workflows import sampler_suite

    captured: dict = {}

    def fake_main(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(sampler_suite.thesis_sampler_suite, "main", fake_main)

    rc = sampler_suite.main(
        seeds=[1, 2],
        n_trials=12,
        datasets=["kdr100"],
        samplers=["qmc"],
        sequential=True,
        n_trials_full=99,
        n_candidates=64,
        autoscale=False,
        execution_profile="thesis_repro",
    )

    assert rc == 0
    assert captured["seeds"] == [1, 2]
    assert captured["n_trials"] == 12
    assert captured["datasets"] == ["kdr100"]
    assert captured["samplers"] == ["qmc"]
    assert captured["sequential"] is True
    assert captured["n_trials_full"] == 99
    assert captured["n_candidates"] == 64
    assert captured["autoscale"] is False
    assert captured["execution_profile"] == "thesis_repro"
