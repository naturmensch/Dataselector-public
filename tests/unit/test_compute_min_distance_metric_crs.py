from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataselector.pipeline.pipeline_utils import compute_min_distance_km


def test_compute_min_distance_uses_metric_projected_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y\n0,0,1,1\n", encoding="utf-8")

    from dataselector.data import io as io_mod

    df = pd.DataFrame({"center_x": [0.0, 1.0, 2.0], "center_y": [0.0, 0.0, 0.0]})
    df.attrs["source_crs"] = "EPSG:3857"
    df.attrs["metric_crs"] = "EPSG:25832"
    df.attrs["transform_applied"] = True
    gdf_metric = pd.DataFrame(
        {"_proj_x": [0.0, 1000.0, 2000.0], "_proj_y": [0.0, 0.0, 0.0]}
    )

    monkeypatch.setattr(io_mod, "load_metadata", lambda *_, **__: df)
    monkeypatch.setattr(io_mod, "get_metric_gdf", lambda *_: gdf_metric)

    min_dist = compute_min_distance_km(str(csv_path))
    assert min_dist == 1.0


def test_compute_min_distance_fails_without_metric_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y\n0,0,1,1\n", encoding="utf-8")

    from dataselector.data import io as io_mod

    df = pd.DataFrame({"center_x": [0.0, 1.0], "center_y": [0.0, 0.0]})
    monkeypatch.setattr(io_mod, "load_metadata", lambda *_, **__: df)
    monkeypatch.setattr(io_mod, "get_metric_gdf", lambda *_: None)

    with pytest.raises(RuntimeError, match="Metric CRS projection missing"):
        compute_min_distance_km(str(csv_path))
