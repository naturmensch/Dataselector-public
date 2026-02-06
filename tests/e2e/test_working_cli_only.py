"""E2E tests for fully-implemented Python CLI commands (no subprocess wrappers).

This test file focuses on CLI commands that have actual Python implementations,
not legacy script wrappers. Tests for wrapper commands (autoscale, xxl, etc.)
are deferred until Phase 3R subprocess wrapper → native Python migration.

Tested commands (fully implemented):
- bootstrap: dataselector bootstrap ...
- optuna-optimize: dataselector optuna-optimize ...
- final-selection: dataselector final-selection ...
- build-tiles: dataselector build-tiles ...
- tools: dataselector tools [check-geo|archive-outputs|list-archives|clean-workspace|...]
"""

import json
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
    """Test tools check-geo works (fully implemented)."""
    result = run_dataselector_cli(["tools", "check-geo"])

    # May fail if GIS deps missing, but shouldn't crash
    assert result.returncode in [0, 1]
    output = (result.stdout.decode() + result.stderr.decode()).lower()
    assert "geopandas" in output or result.returncode == 0


@pytest.mark.integration
def test_tools_clean_workspace_dryrun(tmp_workspace: Path, run_dataselector_cli):
    """Test tools clean-workspace --dry-run (fully implemented)."""
    result = run_dataselector_cli(["tools", "clean-workspace"], cwd=str(tmp_workspace))
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
    """Test bootstrap --help works (fully implemented)."""
    result = run_dataselector_cli(["bootstrap", "--help"])
    assert result.returncode == 0

    output = result.stdout.decode().lower()
    assert "bootstrap" in output or "resampling" in output


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


# Skip tests for wrapper-based commands until Phase 3R migration
@pytest.mark.skip(
    reason="autoscale is a subprocess wrapper (Phase 3R migration pending)"
)
def test_autoscale_skipped(run_dataselector_cli):
    """Test autoscale - SKIPPED until Phase 3R native Python migration."""
    pass


@pytest.mark.skip(reason="xxl is a subprocess wrapper (Phase 3R migration pending)")
def test_xxl_skipped(run_dataselector_cli):
    """Test xxl - SKIPPED until Phase 3R native Python migration."""
    pass


@pytest.mark.skip(
    reason="thesis-pipeline is a subprocess wrapper (Phase 3R migration pending)"
)
def test_thesis_pipeline_skipped(run_dataselector_cli):
    """Test thesis-pipeline - SKIPPED until Phase 3R native Python migration."""
    pass
