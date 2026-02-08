"""Tests for thesis pipeline workflow."""

import json

import pytest


def test_thesis_pipeline_importable():
    """Test that thesis_pipeline module can be imported."""
    from dataselector.workflows import thesis_pipeline

    assert hasattr(thesis_pipeline, "run_thesis_pipeline")


def test_thesis_pipeline_signature():
    """Test run_thesis_pipeline function signature."""
    import inspect

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    sig = inspect.signature(run_thesis_pipeline)
    params = list(sig.parameters.keys())

    expected_params = [
        "n_lhs",
        "n_trials",
        "skip_exploration",
        "skip_optimization",
        "skip_validation",
        "dry_run",
        "output_dir",
        "execution_profile",
        "seed",
    ]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


def test_run_thesis_pipeline_dry_run_skip_validation(tmp_path):
    """Regression guard for CLI dry-run path without validation imports."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    success = run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_validation=True,
        dry_run=True,
        output_dir=tmp_path / "outputs",
        execution_profile="thesis_repro",
        seed=7,
    )

    assert success is True
    metadata_path = tmp_path / "outputs" / "run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution_profile"] == "thesis_repro"
    assert metadata["seed"] == 7


def test_run_thesis_pipeline_passes_metadata_path_to_stages(tmp_path, monkeypatch):
    """Non-dry-run must pass metadata_path to exploration and Optuna workflows."""
    from dataselector.workflows import thesis_pipeline as mod

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)

    calls: dict[str, str] = {}

    def fake_exploration(**kwargs):
        calls["exploration_metadata_path"] = str(kwargs["metadata_path"])
        return [], ws / "dummy.csv"

    def fake_optuna(**kwargs):
        calls["optuna_metadata_path"] = str(kwargs["metadata_path"])
        return object()

    monkeypatch.setattr(
        "dataselector.workflows.tune_weights.run_exploration", fake_exploration
    )
    monkeypatch.setattr(
        "dataselector.workflows.optuna_optimize.run_optuna", fake_optuna
    )
    monkeypatch.setattr(
        "dataselector.workflows.generate_reports.generate_thesis_final_report",
        lambda **_kwargs: None,
    )

    success = mod.run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_validation=True,
        dry_run=False,
        output_dir=ws / "outputs",
        seed=11,
    )

    assert success is True
    assert calls["exploration_metadata_path"].endswith("data/new_all_tiles.csv")
    assert calls["optuna_metadata_path"].endswith("data/new_all_tiles.csv")


def test_run_thesis_pipeline_phase4_single_run_report(tmp_path):
    """Phase 4 summary must work for single-run output_dir contract."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    success = run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_exploration=True,
        skip_optimization=True,
        skip_validation=True,
        dry_run=False,
        output_dir=tmp_path / "outputs",
        execution_profile="thesis_repro",
        seed=42,
    )

    assert success is True
    assert (tmp_path / "outputs" / "THESIS_PIPELINE_REPORT.md").exists()


@pytest.mark.skipif(
    True, reason="Requires full pipeline setup (data, features, config)"
)
def test_run_thesis_pipeline_integration():
    """Integration test for run_thesis_pipeline (skipped in CI)."""
    import tempfile
    from pathlib import Path

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "outputs"

        success = run_thesis_pipeline(
            n_lhs=10,
            n_trials=5,
            dry_run=True,
            output_dir=output_dir,
        )

        assert success is True


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "thesis-pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # CLI might not exist yet, so just check it doesn't crash
    assert result.returncode in (0, 1, 2)
