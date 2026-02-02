import pandas as pd
import pytest

from dataselector.data.metadata_processor import MetadataProcessor


def test_calculate_spatial_distance_degree_inputs_with_gdf_metric_falls_back_to_haversine():
    mp = MetadataProcessor("")
    mp.gdf_metric = pd.DataFrame({"_proj_x": [0], "_proj_y": [0]})

    # 1 degree longitude at equator ≈ 111.195 km
    d = mp.calculate_spatial_distance(0.0, 0.0, 0.0, 1.0)
    assert pytest.approx(d, rel=1e-3) == 111.195


def test_calculate_spatial_distance_projected_inputs_use_euclid():
    mp = MetadataProcessor("")
    mp.gdf_metric = pd.DataFrame({"_proj_x": [0], "_proj_y": [0]})

    # inputs in meters (x2 - x1 = 1000m) → 1.0 km
    d = mp.calculate_spatial_distance(0.0, 0.0, 0.0, 1000.0)
    assert pytest.approx(d, rel=1e-6) == 1.0
