from pathlib import Path

import pandas as pd
import pytest

from dataselector.data.io import load_metadata


def _write_minimal_metadata(csv_path: Path, include_image_path: bool = True) -> None:
    data = {
        "longName": ["KDR_001_Test_1901.png", "KDR_002_Test_1902.png"],
        "ul_x": [499950.0, 500050.0],
        "ul_y": [5900050.0, 5900150.0],
        "lr_x": [500050.0, 500150.0],
        "lr_y": [5899950.0, 5900050.0],
        "year": [1901, 1902],
    }
    if include_image_path:
        data["image_path"] = ["data/images/KDR_001_Test_1901.png", None]
    pd.DataFrame(data).to_csv(csv_path, index=False)


def test_load_metadata_keeps_existing_image_paths_when_image_dir_missing(tmp_path):
    csv_path = tmp_path / "meta.csv"
    _write_minimal_metadata(csv_path, include_image_path=True)
    missing_dir = tmp_path / "does-not-exist"

    df = load_metadata(csv_path, image_dir=missing_dir)

    assert df.loc[0, "image_path"] == "data/images/KDR_001_Test_1901.png"
    assert df.loc[1, "image_path"] == "missing_placeholder.png"


def test_load_metadata_sets_placeholder_without_image_path_and_missing_dir(tmp_path):
    csv_path = tmp_path / "meta.csv"
    _write_minimal_metadata(csv_path, include_image_path=False)
    missing_dir = tmp_path / "does-not-exist"

    df = load_metadata(csv_path, image_dir=missing_dir)

    assert "image_path" in df.columns
    assert (df["image_path"] == "missing_placeholder.png").all()


def test_load_metadata_strict_resolution_raises_when_image_dir_missing(tmp_path):
    csv_path = tmp_path / "meta.csv"
    _write_minimal_metadata(csv_path, include_image_path=False)
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match="Image directory does not exist"):
        load_metadata(
            csv_path,
            image_dir=missing_dir,
            strict_image_resolution=True,
        )
