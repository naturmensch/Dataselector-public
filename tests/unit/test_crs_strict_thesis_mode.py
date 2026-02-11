from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_load_metadata_strict_metric_crs_fails_when_metric_projection_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from dataselector.data import io as io_mod

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")

    class _DummyProcessor:
        def __init__(self, _csv_path: str):
            self.source_crs = "EPSG:3857"
            self.metric_crs = None
            self.transform_applied = False

        def load_csv(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "ul_x": [1.0],
                    "ul_y": [2.0],
                    "lr_x": [3.0],
                    "lr_y": [4.0],
                    "year": [1900],
                    "longName": ["img_0.png"],
                }
            )

        def add_temporal_metadata(self) -> pd.DataFrame:
            return self.load_csv()

        def ensure_metric_crs(self, target_epsg: int = 25832, strict: bool = False):
            return None

    monkeypatch.setattr(io_mod, "MetadataProcessor", _DummyProcessor)

    with pytest.raises(
        RuntimeError, match="Strict CRS mode requires metric reprojection"
    ):
        io_mod.load_metadata(csv_path, resolve_images=False, strict_metric_crs=True)


def test_load_metadata_non_strict_allows_missing_metric_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from dataselector.data import io as io_mod

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")

    class _DummyProcessor:
        def __init__(self, _csv_path: str):
            self.source_crs = "EPSG:3857"
            self.metric_crs = None
            self.transform_applied = False

        def load_csv(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "ul_x": [1.0],
                    "ul_y": [2.0],
                    "lr_x": [3.0],
                    "lr_y": [4.0],
                    "year": [1900],
                    "longName": ["img_0.png"],
                }
            )

        def add_temporal_metadata(self) -> pd.DataFrame:
            return self.load_csv()

        def ensure_metric_crs(self, target_epsg: int = 25832, strict: bool = False):
            return None

    monkeypatch.setattr(io_mod, "MetadataProcessor", _DummyProcessor)
    df = io_mod.load_metadata(csv_path, resolve_images=False, strict_metric_crs=False)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs.get("metric_crs") is None
