import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_metrics


def test_compute_metrics_uses_projected_coords_when_available():
    df = pd.DataFrame({"N": [0.0, 0.0], "left": [0.0, 0.0], "year": [2000, 2000]})
    gdf_metric = pd.DataFrame({"_proj_x": [0.0, 1000.0], "_proj_y": [0.0, 0.0]}, index=df.index)
    # attach projected dataframe as attribute like load_metadata does
    df.gdf_metric = gdf_metric

    metrics = compute_metrics([0, 1], df, np.array([0, 1]), np.zeros((2, 4)))

    assert pytest.approx(metrics["spatial_mean_km"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["spatial_min_km"], rel=1e-6) == 1.0
