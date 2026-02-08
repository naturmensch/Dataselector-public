from pathlib import Path

FIXTURE_DIR = Path("tests/fixtures/real_tiles")
MAX_TOTAL_BYTES = 120 * 1024 * 1024  # 120 MB hard ceiling for CI fixture tier


def test_real_tiles_fixture_shape_and_size():
    assert FIXTURE_DIR.exists(), "real_tiles fixture directory is missing"

    files = sorted(p for p in FIXTURE_DIR.iterdir() if p.is_file())
    png_files = [p for p in files if p.suffix.lower() == ".png"]
    aux_files = [p for p in files if p.name.endswith(".png.aux.xml")]

    # Policy: keep this fixture as a tiny fixed baseline (5 tiles + 5 aux xml).
    assert len(png_files) == 5, "Expected exactly 5 PNG fixture tiles"
    assert len(aux_files) == 5, "Expected exactly 5 aux.xml sidecar files"

    total_bytes = sum(p.stat().st_size for p in files)
    assert (
        total_bytes <= MAX_TOTAL_BYTES
    ), f"real_tiles fixture exceeded {MAX_TOTAL_BYTES} bytes policy limit"
