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
    # check parsed coordinates
    assert df.loc[0, "N"] == 52.1

    prov = out_csv.parent / "new_all_tiles_provenance.json"
    assert prov.exists()
    prov_json = json.loads(prov.read_text())
    assert prov_json["rows"] == 2
