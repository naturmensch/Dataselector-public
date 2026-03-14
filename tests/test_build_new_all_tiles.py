import json
from pathlib import Path

import pandas as pd

from dataselector.data.build_tiles import build_tiles


def test_build_tiles_scans_images_and_sidecars(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()

    img1 = images / "KDR_001_Someplace_1901.png"
    img1.write_bytes(b"png")
    img2 = images / "KDR_002_Other_1950.jpg"
    img2.write_bytes(b"jpg")

    # Sidecar XML for first image only.
    (images / "KDR_001_Someplace_1901.aux.xml").write_text(
        "<root><lat>52.1</lat><lon>13.4</lon></root>"
    )

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(image_dir=images, out=out_csv)
    assert rc == 0
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert "image_filename" in df.columns
    row1 = df[df["image_filename"] == "KDR_001_Someplace_1901.png"].iloc[0]
    assert float(row1["lat"]) == 52.1
    assert float(row1["lon"]) == 13.4

    prov_path = out_csv.parent / "new_all_tiles_provenance.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["rows"] == 2
    assert prov["source"] == "images_scanned"


def test_build_tiles_force_source_updates_provenance(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "KDR_003_New_2000.png").write_bytes(b"png")
    (images / "KDR_003_New_2000.aux.xml").write_text(
        "<root><lat>50.0</lat><lon>10.0</lon></root>"
    )

    force_source = tmp_path / "all_tiles.csv"
    pd.DataFrame({"id": [1], "dummy": ["x"]}).to_csv(force_source, index=False)

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(image_dir=images, out=out_csv, force_source=str(force_source))
    assert rc == 0

    prov_path = out_csv.parent / "new_all_tiles_provenance.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["source"] == "all_tiles.csv"
    assert prov["source_sha256"]


def test_build_tiles_enriches_city_from_force_source_longname(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "KDR_146.png").write_bytes(b"png")

    force_source = tmp_path / "legacy_epsg_source.csv"
    pd.DataFrame(
        {
            "shortName": ["KDR_146.png", "KDR_001.png"],
            "longName": ["KDR_146_Hamburg_1918.png", "KDR_001_Kiel_1917.png"],
        }
    ).to_csv(force_source, index=False)

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(image_dir=images, out=out_csv, force_source=str(force_source))
    assert rc == 0

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["shortName"] == "KDR_146"
    assert row["longName"] == "KDR_146_Hamburg_1918.png"
    assert row["city"] == "Hamburg"
    assert row["city_source"] == "longname_parse"
    assert int(row["year"]) == 1918


def test_build_tiles_normalizes_shortname_case_and_suffix(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "KDR_079a.png").write_bytes(b"png")

    force_source = tmp_path / "legacy_epsg_source.csv"
    pd.DataFrame(
        {
            "shortName": ["KDR_079A.png"],
            "longName": ["KDR_079a_Helgoland_1918.png"],
        }
    ).to_csv(force_source, index=False)

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(
        image_dir=images,
        out=out_csv,
        force_source=str(force_source),
        name_source_csv=str(force_source),
    )
    assert rc == 0

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["shortName"] == "KDR_079a"
    assert row["city"] == "Helgoland"
    assert row["city_source"] in {"longname_parse", "epsg_source", "variant_base"}


def test_build_tiles_applies_manual_city_overrides(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "KDR_046.png").write_bytes(b"png")

    force_source = tmp_path / "legacy_epsg_source.csv"
    pd.DataFrame({"shortName": [], "longName": []}).to_csv(force_source, index=False)

    overrides = tmp_path / "city_overrides.csv"
    pd.DataFrame(
        {
            "shortName": ["KDR_046"],
            "city": ["Karthaus"],
            "source": ["manual_validation"],
            "note": ["test override"],
        }
    ).to_csv(overrides, index=False)

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(
        image_dir=images,
        out=out_csv,
        force_source=str(force_source),
        name_source_csv=str(force_source),
        city_overrides=str(overrides),
    )
    assert rc == 0

    df = pd.read_csv(out_csv)
    row = df.iloc[0]
    assert row["shortName"] == "KDR_046"
    assert row["city"] == "Karthaus"
    assert row["city_source"] == "manual_override"


def test_build_tiles_persists_explicit_sidecar_crs(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "KDR_010_Test_1901.png").write_bytes(b"png")
    (images / "KDR_010_Test_1901.png.aux.xml").write_text(
        "<PAMDataset><SRS>EPSG:3857</SRS><GeoTransform>0,1,0,0,0,-1</GeoTransform></PAMDataset>",
        encoding="utf-8",
    )

    out_csv = tmp_path / "new_all_tiles.csv"
    rc = build_tiles(image_dir=images, out=out_csv)
    assert rc == 0

    df = pd.read_csv(out_csv)
    row = df.iloc[0]
    assert row["source_crs"] == "EPSG:3857"
    assert row["crs_source"] == "sidecar_xml"
    assert row["crs_provenance"] == "explicit_sidecar_xml"
    assert bool(row["crs_explicit"]) is True
