import numpy as np
import pandas as pd

from dataselector.selection.multi_criteria_facility_location import (
    MultiCriteriaFacilityLocation,
)
from dataselector.selection.spatial_facility_location import (
    SpatialConstrainedFacilityLocation,
)


class DummyMetadata:
    def __init__(self, latitudes, longitudes, years, proj_x=None, proj_y=None):
        self._df = pd.DataFrame({"N": latitudes, "left": longitudes, "year": years})
        self.gdf_metric = None
        if proj_x is not None and proj_y is not None:
            self.gdf_metric = pd.DataFrame({"_proj_x": proj_x, "_proj_y": proj_y})

    def __getitem__(self, key):
        return self._df[key]


def test_multi_criteria_uses_projected_coords():
    # Create three points: (0,0), (0,100000), (0,200000) meters apart in Y
    latitudes = [0.0, 0.0, 0.0]
    longitudes = [0.0, 0.0, 0.0]
    years = [1900, 1910, 1920]

    proj_x = [0.0, 0.0, 0.0]
    proj_y = [0.0, 100000.0, 200000.0]

    meta = DummyMetadata(latitudes, longitudes, years, proj_x=proj_x, proj_y=proj_y)
    m = MultiCriteriaFacilityLocation(
        n_samples=2,
        metadata=meta,
        alpha_visual=0.5,
        beta_spatial=0.25,
        gamma_temporal=0.25,
        min_distance_km=50.0,
    )

    X = np.random.RandomState(0).randn(3, 4)  # dummy visual features
    _dcombined = m._compute_pairwise_distances(X)

    # Check that spatial km matrix was set and equals approx expected distances (100 km steps)
    sp_km = m._spatial_km
    assert sp_km.shape == (3, 3)
    assert abs(sp_km[0, 1] - 100.0) < 1e-6
    assert abs(sp_km[1, 2] - 100.0) < 1e-6


def test_spatial_selector_respects_min_distance():
    # Two close points and one far point
    latitudes = [0.0, 0.0, 0.0]
    longitudes = [0.0, 0.0, 0.0]
    years = [1900, 1900, 1900]
    proj_x = [0.0, 0.0, 50000.0]
    proj_y = [0.0, 40000.0, 0.0]

    meta = DummyMetadata(latitudes, longitudes, years, proj_x=proj_x, proj_y=proj_y)

    sel = SpatialConstrainedFacilityLocation(
        n_samples=2, metadata=meta, min_distance_km=50.0
    )
    # features to pass (not used for spatial constraint logic heavily)
    X = np.zeros((3, 2))
    # When fitting, the selector must avoid selecting the two close points
    sel.fit(X)
    # Ensure that selected indices are at least min_distance apart
    ranked = sel.ranking
    assert len(ranked) == 2
    # compute pairwise distances between selected using projected coords
    a, b = ranked[0], ranked[1]
    dx = proj_x[a] - proj_x[b]
    dy = proj_y[a] - proj_y[b]
    dist_km = (dx * dx + dy * dy) ** 0.5 / 1000.0
    assert dist_km >= 50.0 - 1e-6
