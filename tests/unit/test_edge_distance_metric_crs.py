from __future__ import annotations

import pandas as pd

from dataselector.data.spatial_distance import edge_distance_km, tile_bounds_to_metric


def test_edge_distance_is_zero_for_touching_tiles() -> None:
    metadata = pd.DataFrame(
        {
            "ul_x": [0.0, 1000.0],
            "ul_y": [1000.0, 1000.0],
            "lr_x": [1000.0, 2000.0],
            "lr_y": [0.0, 0.0],
        }
    )
    metadata.attrs["source_crs"] = "EPSG:25832"
    metric = tile_bounds_to_metric(metadata, target_epsg=25832, strict=True)
    d = edge_distance_km(
        metric.loc[0, "_minx_m"],
        metric.loc[0, "_maxx_m"],
        metric.loc[0, "_miny_m"],
        metric.loc[0, "_maxy_m"],
        metric.loc[1, "_minx_m"],
        metric.loc[1, "_maxx_m"],
        metric.loc[1, "_miny_m"],
        metric.loc[1, "_maxy_m"],
    )
    assert d == 0.0
