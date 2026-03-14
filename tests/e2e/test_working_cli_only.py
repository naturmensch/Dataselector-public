"""E2E tests for current canonical CLI contracts."""

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_dataselector_help(run_dataselector_cli):
    """Test dataselector --help works."""
    result = run_dataselector_cli(["--help"])
    assert result.returncode == 0
    assert (
        "usage:" in result.stdout.decode().lower()
        or "dataselector" in result.stdout.decode()
    )


@pytest.mark.smoke
def test_build_tiles_help(run_dataselector_cli):
    """Test build-tiles --help works (fully implemented)."""
    result = run_dataselector_cli(["build-tiles", "--help"])
    assert result.returncode == 0
    assert "--image-dir" in result.stdout.decode()


@pytest.mark.integration
def test_tools_check_geo(run_dataselector_cli):
    """Test check-geo works with top-level CLI command."""
    result = run_dataselector_cli(["check-geo"])

    # May fail if GIS deps missing, but shouldn't crash
    assert result.returncode in [0, 2]
    output = (result.stdout.decode() + result.stderr.decode()).lower()
    assert "geopandas" in output or result.returncode == 0


@pytest.mark.integration
def test_tools_clean_workspace_dryrun(tmp_workspace: Path, run_dataselector_cli):
    """Test clean-workspace dry-run default behavior."""
    result = run_dataselector_cli(["clean-workspace"], cwd=str(tmp_workspace))
    # Dry-run is default, should succeed
    assert result.returncode == 0


@pytest.mark.error
def test_invalid_subcommand(run_dataselector_cli):
    """Test invalid subcommand fails gracefully."""
    result = run_dataselector_cli(["nonexistent-command"])
    # Should fail but not crash
    assert result.returncode != 0

    stderr = result.stderr.decode().lower()
    assert "error" in stderr or "unrecognized" in stderr


@pytest.mark.smoke
def test_build_tiles_empty_directory(tmp_workspace: Path, run_dataselector_cli):
    """Test build-tiles with empty image directory (fully implemented)."""
    image_dir = tmp_workspace / "empty_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    output_csv = tmp_workspace / "tiles.csv"

    result = run_dataselector_cli(
        ["build-tiles", "--image-dir", str(image_dir), "--out", str(output_csv)],
        cwd=str(tmp_workspace),
    )

    # Should complete successfully (empty CSV is OK)
    assert result.returncode == 0, f"build-tiles failed:\n{result.stderr.decode()}"
    assert output_csv.exists(), f"Output CSV {output_csv} not created"


@pytest.mark.integration
def test_bootstrap_help(run_dataselector_cli):
    """Test current bootstrap commands expose --help."""
    result_final = run_dataselector_cli(["bootstrap-final", "--help"])
    assert result_final.returncode == 0
    assert "bootstrap" in result_final.stdout.decode().lower()

    result_pareto = run_dataselector_cli(["bootstrap-pareto", "--help"])
    assert result_pareto.returncode == 0
    assert "pareto" in result_pareto.stdout.decode().lower()


@pytest.mark.integration
def test_final_selection_help(run_dataselector_cli):
    """Test final-selection --help works (fully implemented)."""
    result = run_dataselector_cli(["final-selection", "--help"])
    assert result.returncode == 0


@pytest.mark.integration
def test_optuna_optimize_help(run_dataselector_cli):
    """Test optuna-optimize --help works (fully implemented)."""
    result = run_dataselector_cli(["optuna-optimize", "--help"])
    assert result.returncode == 0

    output = result.stdout.decode().lower()
    assert "optuna" in output or "optimization" in output


@pytest.mark.integration
def test_autoscale_help(run_dataselector_cli):
    """Test autoscale --help works for the active CLI surface."""
    result = run_dataselector_cli(["autoscale", "--help"])
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    assert "--n-trials" in out
    assert "--stages" in out


@pytest.mark.integration
def test_xxl_help(run_dataselector_cli):
    """Test xxl --help still works as a secondary-active CLI path."""
    result = run_dataselector_cli(["xxl", "--help"])
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    assert "--phase" in out


@pytest.mark.integration
def test_thesis_pipeline_help(run_dataselector_cli):
    """Test thesis-pipeline --help works for the canonical CLI path."""
    result = run_dataselector_cli(["thesis-pipeline", "--help"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "--use-params" in out
