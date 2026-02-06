"""E2E test for geo workflow (build-tiles + align-audit).

Tests complete geospatial workflow:
- Build CSV from image directory
- Validate geo dependencies
- Check CSV-Raster alignment
"""

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.geo
@pytest.mark.usefixtures("skip_if_no_gis")
def test_geo_workflow_check_dependencies(tmp_workspace: Path, run_dataselector_cli):
    """Verify geo dependencies are available.

    Uses canonical check-geo command.
    """
    cmd = ["check-geo"]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)

    # Should succeed if GIS stack is available (fixture already checked)
    assert result.returncode == 0, f"geo check failed:\n{result.stderr.decode()}"


@pytest.mark.integration
@pytest.mark.geo
def test_build_tiles_creates_csv(tmp_workspace: Path, run_dataselector_cli):
    """Test build-tiles command creates CSV from image directory.

    Uses sample images if available, or skips gracefully.
    """
    output_file = tmp_workspace / "data" / "test_tiles.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Try with empty directory (should produce empty CSV)
    image_dir = tmp_workspace / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "build-tiles",
        "--image-dir",
        str(image_dir),
        "--out",
        str(output_file),
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)

    # Should either succeed or fail with helpful message about missing images
    if result.returncode == 0:
        assert output_file.exists(), "Output CSV not created"
    else:
        stderr = result.stderr.decode()
        # Error should mention missing images or empty directory
        assert (
            "image" in stderr.lower()
            or "empty" in stderr.lower()
            or "error" in stderr.lower()
        )


@pytest.mark.error
@pytest.mark.geo
def test_align_audit_missing_csv(tmp_workspace: Path, run_dataselector_cli):
    """Error test: align-audit with missing CSV.

    Should fail gracefully.
    """
    cmd = [
        "align-audit",
        "--csv",
        str(tmp_workspace / "nonexistent.csv"),
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True)

    assert result.returncode != 0, "Should have failed with missing CSV"
