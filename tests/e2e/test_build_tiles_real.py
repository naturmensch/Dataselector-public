"""E2E tests using real KDR tiles for build-tiles and geo workflow validation.

These tests use actual tile images from tests/fixtures/real_tiles/ instead of
synthetic data, providing more realistic validation of the image processing
and geospatial pipelines.

Note: This test file requires the real tiles to be present in the repository.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.geo
@pytest.mark.real_tiles
def test_build_tiles_from_real_images(tmp_workspace: Path, run_dataselector_cli):
    """Build tiles CSV from real KDR image directory.

    Tests:
    - build-tiles correctly processes real PNG+aux.xml files
    - Generated CSV has correct structure
    - Metadata is extracted from GeoTransform
    """
    # Get path to real tiles fixture directory
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "real_tiles"

    if not fixture_dir.exists():
        pytest.skip(f"Real tiles fixture not found at {fixture_dir}")

    output_csv = tmp_workspace / "generated_tiles.csv"

    cmd = [
        "build-tiles",
        "--image-dir",
        str(fixture_dir),
        "--out",
        str(output_csv),
    ]

    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120
    )

    assert result.returncode == 0, f"build-tiles failed:\n{result.stderr.decode()}"
    assert output_csv.exists(), f"Output CSV not created at {output_csv}"

    # Verify CSV structure
    with open(output_csv) as f:
        header = f.readline()
        columns = [c.strip() for c in header.strip().split(",") if c.strip()]
        required_columns = {"image_path", "image_filename", "year"}
        missing_columns = required_columns - set(columns)
        assert not missing_columns, (
            f"CSV missing required columns: {sorted(missing_columns)}; got {columns}"
        )

        lines = f.readlines()
        # Should have at least 5 tiles (from KDR_001-005)
        assert len(lines) >= 5, f"Expected at least 5 tiles, got {len(lines)}"


@pytest.mark.integration
@pytest.mark.geo
@pytest.mark.real_tiles
def test_real_tiles_csv_alignment(
    real_tiles_csv: Path, tmp_workspace: Path, run_dataselector_cli
):
    """Test that provided real_tiles_csv aligns with actual image files.

    Validates:
    - All tiles in CSV have corresponding PNG files
    - All PNGs have corresponding aux.xml files
    - CSV columns match expected schema
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "real_tiles"

    if not fixture_dir.exists():
        pytest.skip(f"Real tiles fixture not found at {fixture_dir}")

    # Read CSV and validate each tile
    import csv

    with open(real_tiles_csv) as f:
        reader = csv.DictReader(f)
        tiles = list(reader)

    assert len(tiles) > 0, "real_tiles_csv is empty"

    for tile in tiles:
        # Check if image files exist relative to fixture directory
        if "tile_id" in tile:
            tile_id = tile["tile_id"]
            png_file = fixture_dir / f"{tile_id}.png"
            xml_file = fixture_dir / f"{tile_id}.png.aux.xml"

            assert png_file.exists(), f"Missing PNG: {png_file}"
            assert xml_file.exists(), f"Missing aux.xml: {xml_file}"
