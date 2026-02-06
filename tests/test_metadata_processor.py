from pathlib import Path

import pytest

# Import the module directly to avoid importing heavy top-level deps from package
REPO_ROOT = Path(__file__).resolve().parents[1]
from tests.utils import load_module_from_path

_mp_mod = load_module_from_path("test_metadata_processor_mod", REPO_ROOT / "dataselector" / "data" / "metadata_processor.py")
MetadataProcessor = _mp_mod.MetadataProcessor


def write_csv(tmp_path, content: str) -> str:
    p = tmp_path / "test_meta.csv"
    p.write_text(content)
    return str(p)


def test_load_and_standardize_csv(tmp_path):
    csv = """File,N,left\nKDR_001_Someplace_1901.png,52,13.4\nKDR_002_Other_1950.png,52.1,13.5\n"""
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    df = proc.load_csv()
    assert "longName" in df.columns
    assert df["longName"].iloc[0] == "KDR_001_Someplace_1901.png"
    assert isinstance(df["N"].iloc[0], float)


def test_extract_year_and_temporal(tmp_path):
    csv = """File,N,left\nKDR_001_Someplace_1901.png,52,13.4\nKDR_002_Other_1950.png,52.1,13.5\n"""
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    assert proc.extract_year("KDR_001_Whatever_1901.png") == 1901
    proc.add_temporal_metadata()
    assert "year" in proc.df.columns
    assert list(proc.df["year"]) == [1901, 1950]


def test_spatial_distance_and_filter(tmp_path):
    csv = """File,N,left\nA.png,0,0\nB.png,0,1\nC.png,50,50\n"""
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    # distance between (0,0) and (0,1) should be ~111 km
    d = proc.calculate_spatial_distance(0, 0, 0, 1)
    assert pytest.approx(111, rel=0.01) == d
    # apply filter with min_distance > 100 should exclude second tile
    valid = proc.apply_spatial_filter(min_distance_km=120)
    assert 0 in valid
    assert len(valid) < len(proc.df)


def test_resolve_image_paths_prefers_shortname(tmp_path):
    csv_lines = [
        "shortName,longName,N,left",
        "IMG1,KDR_001_IMG1_1901.png,52,13",
        ",KDR_002_IMG2_1950.png,52,14",
    ]
    csv = "\n".join(csv_lines) + "\n"
    p = tmp_path / "imgdir"
    p.mkdir()
    # create files
    (p / "IMG1.png").write_text("x")
    (p / "kdr_002_img2_1950.png").write_text("x")
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    df = proc.resolve_image_paths(str(p), prefer_shortname=True)
    # image paths resolved and filenames present
    assert df["image_path"].iloc[0] is not None
    assert df["image_filename"].iloc[0] == "IMG1.png"
    assert df["image_path"].iloc[1] is not None


def test_convert_dbf_to_csv_raises_for_non_dbf(tmp_path):
    csv = "File,N,left\nA.png,0,0\n"
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    with pytest.raises(ValueError):
        proc.convert_dbf_to_csv()


def test_metadata_standard_flow_equivalent(tmp_path):
    # From previous 'new' test suite: ensure standard flow and temporal metadata behavior
    csv = """File,N,left
KDR_001_Someplace_1901.png,52,13.4
KDR_002_Other_1950.png,52.1,13.5
"""
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    df = proc.load_csv()
    assert "longName" in df.columns
    proc.add_temporal_metadata()
    assert list(df["year"]) == [1901, 1950]


def test_resolve_image_paths_equivalent(tmp_path):
    csv = """shortName,longName,N,left
IMG1,KDR_001_IMG1_1901.png,52,13
,KDR_002_IMG2_1950.png,52,14
"""
    imgdir = tmp_path / "imgdir"
    imgdir.mkdir()
    (imgdir / "IMG1.png").write_text("x")
    (imgdir / "kdr_002_img2_1950.png").write_text("x")
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    df = proc.resolve_image_paths(str(imgdir), prefer_shortname=True)
    assert df["image_filename"].iloc[0] == "IMG1.png"
    assert df["image_path"].iloc[1] is not None
