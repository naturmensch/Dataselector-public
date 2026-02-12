from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from dataselector.selection.multi_criteria_facility_location import (
    MultiCriteriaFacilityLocation,
)


def _touching_tiles_metadata() -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "ul_x": [0.0, 1000.0],
            "ul_y": [1000.0, 1000.0],
            "lr_x": [1000.0, 2000.0],
            "lr_y": [0.0, 0.0],
            "center_x": [500.0, 1500.0],
            "center_y": [500.0, 500.0],
            "year": [1900, 1901],
        }
    )
    meta.attrs["source_crs"] = "EPSG:25832"
    return meta


def test_edge_constraint_blocks_touching_neighbor_when_min_distance_positive() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    selector = MultiCriteriaFacilityLocation(
        n_samples=2,
        metadata=_touching_tiles_metadata(),
        alpha_visual=0.7,
        beta_spatial=0.15,
        gamma_temporal=0.15,
        min_distance_km=0.01,
        spatial_constraint_metric="edge_to_edge_km",
    )
    selector.fit(features)
    assert selector.ranking is not None
    assert len(selector.ranking) == 1


def test_edge_constraint_can_be_disabled_with_zero_min_distance() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    selector = MultiCriteriaFacilityLocation(
        n_samples=2,
        metadata=_touching_tiles_metadata(),
        alpha_visual=0.7,
        beta_spatial=0.15,
        gamma_temporal=0.15,
        min_distance_km=0.0,
        spatial_constraint_metric="edge_to_edge_km",
    )
    selector.fit(features)
    assert selector.ranking is not None
    assert len(selector.ranking) == 2
