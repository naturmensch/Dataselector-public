import json
import subprocess
from pathlib import Path

import pandas as pd


def test_build_new_all_tiles_tmpdir(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()

    # create two fake image files
    img1 = images / "KDR_001_Someplace_1901.png"
    img1.write_bytes(b"png")
    img2 = images / "KDR_002_Other_1950.jpg"
    img2.write_bytes(b"jpg")

    # create sidecar for img1 with simple XML containing coordinates
    sc1 = images / "KDR_001_Someplace_1901.aux.xml"
    sc1.write_text("""<root><lat>52.1</lat><lon>13.4</lon></root>""")

    # img2 without sidecar

    out_csv = tmp_path / "new_all_tiles.csv"

    # run script
    import sys
    cmd = [sys.executable, "scripts/build_new_all_tiles.py", "--image-dir", str(images), "--out", str(out_csv)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + "\n" + res.stderr

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    # Expect two rows and columns
    assert len(df) == 2
    assert "longName" in df.columns
    assert df.loc[0, "longName"] == "KDR_001_Someplace_1901.png"
    # check parsed coordinates (now converted to EPSG:3857)
    # lat 52.1, lon 13.4 -> approx N: 6818227, left: 1491681
    assert abs(df.loc[0, "N"] - 6818227) < 1
    assert abs(df.loc[0, "left"] - 1491681) < 1

    prov = out_csv.parent / "new_all_tiles_provenance.json"
    assert prov.exists()
    prov_json = json.loads(prov.read_text())
    assert prov_json["rows"] == 2


def test_build_new_all_tiles_with_base_merge(tmp_path: Path):
    # Create base CSV in tmp_path and copy to data/ for the script to find it
    base_csv = tmp_path / "KDR100_foliage_with_files_epsg3857.csv"
    base_df = pd.DataFrame({
        "longName": ["KDR_001_Someplace_1901.png", "KDR_002_Other_1950.jpg", "KDR_003_New_2000.png"],
        "N": [5800000.0, 5900000.0, None],  # EPSG:3857 coords (meters)
        "left": [1400000.0, 1500000.0, None],
        "shortName": ["KDR_001", "KDR_002", "KDR_003"],
        "year": [1901, 1950, 2000]
    })
    base_df.to_csv(base_csv, index=False)

    # Copy to data/ so script finds it
    import shutil
    data_base = Path("data") / "KDR100_foliage_with_files_epsg3857.csv"
    shutil.copy(base_csv, data_base)

    try:
        images = tmp_path / "images"
        images.mkdir()

        # Create image files
        img1 = images / "KDR_001_Someplace_1901.png"
        img1.write_bytes(b"png")
        img2 = images / "KDR_002_Other_1950.jpg"
        img2.write_bytes(b"jpg")
        img3 = images / "KDR_003_New_2000.png"  # New image not in base
        img3.write_bytes(b"png")

        # Sidecar for img1 with lat/lon (should NOT overwrite base)
        sc1 = images / "KDR_001_Someplace_1901.aux.xml"
        sc1.write_text("""<root><lat>52.1</lat><lon>13.4</lon></root>""")

        # Sidecar for img3 (new, not in base)
        sc3 = images / "KDR_003_New_2000.aux.xml"
        sc3.write_text("""<root><lat>50.0</lat><lon>10.0</lon></root>""")

        out_csv = tmp_path / "new_all_tiles.csv"

        # Run script
        import sys
        cmd = [sys.executable, "scripts/build_new_all_tiles.py", "--image-dir", str(images), "--out", str(out_csv)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + "\n" + res.stderr

        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        # Expect 3 rows (all from base, enriched with images)
        assert len(df) == 3
        assert "longName" in df.columns
        assert "N" in df.columns
        assert "left" in df.columns

        # Check that base values are preserved (no overwrite)
        row1 = df[df["longName"] == "KDR_001_Someplace_1901.png"].iloc[0]
        assert row1["N"] == 5800000.0  # Base value preserved
        assert row1["left"] == 1400000.0

        # Check img3 got coordinates from sidecar (converted to EPSG:3857)
        row3 = df[df["longName"] == "KDR_003_New_2000.png"].iloc[0]
        # Approximate EPSG:3857 for 50.0, 10.0
        expected_n = 6446275.841  # Approx for lat 50
        expected_left = 1113194.908  # Approx for lon 10
        assert abs(row3["N"] - expected_n) < 1000  # Allow some tolerance
        assert abs(row3["left"] - expected_left) < 1000

        # Check img2 has base values
        row2 = df[df["longName"] == "KDR_002_Other_1950.jpg"].iloc[0]
        assert row2["N"] == 5900000.0
        assert row2["left"] == 1500000.0
    finally:
        # Clean up
        if data_base.exists():
            data_base.unlink()
