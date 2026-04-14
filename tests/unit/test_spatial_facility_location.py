from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("apricot")

from dataselector.selection.spatial_facility_location import (
    haversine_distance,
    haversine_matrix,
)


def test_haversine_matrix_shape_symmetry_and_zero_diagonal() -> None:
    lats = np.array([52.52, 48.14, 53.55], dtype=float)
    lons = np.array([13.405, 11.58, 10.0], dtype=float)

    d = haversine_matrix(lats, lons)

    assert d.shape == (3, 3)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)


def test_haversine_matrix_matches_pairwise_distance_function() -> None:
    lats = np.array([52.52, 48.14], dtype=float)
    lons = np.array([13.405, 11.58], dtype=float)

    d = haversine_matrix(lats, lons)
    expected = haversine_distance(lats[0], lons[0], lats[1], lons[1])

    assert d[0, 1] == pytest.approx(expected, rel=1e-10)
    assert d[1, 0] == pytest.approx(expected, rel=1e-10)


def test_haversine_distance_known_city_pair_range() -> None:
    # Berlin (52.52, 13.405) to Munich (48.14, 11.58) is ~504 km.
    km = haversine_distance(52.52, 13.405, 48.14, 11.58)
    assert 500.0 < km < 510.0
