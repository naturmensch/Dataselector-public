import numpy as np
import pandas as pd
from src.multi_criteria_facility_location import MultiCriteriaFacilityLocation


def make_metadata(latlon_pairs, years):
    df = pd.DataFrame({"N": [p[0] for p in latlon_pairs], "left": [p[1] for p in latlon_pairs], "year": years})
    return df


def test_spatial_penalty_increases_nearby_distances():
    # Create three candidates: A and B are close, C is far
    latlon = [(0.0, 0.0), (0.01, 0.01), (1.0, 1.0)]  # small delta ~1.5km for second
    years = [1900, 1900, 1950]
    meta = make_metadata(latlon, years)

    X = np.zeros((3, 4))  # dummy features (visual equal)

    # Without penalty
    m0 = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.0)
    d0 = m0._compute_pairwise_distances(X)

    # With penalty
    m1 = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.2)
    d1 = m1._compute_pairwise_distances(X)

    # penalty should increase distances between nearby pairs (0 and 1)
    assert d1[0, 1] >= d0[0, 1]
    # distant pair should be unchanged
    assert np.isclose(d1[0, 2], d0[0, 2], atol=1e-8)


def test_soft_penalty_allows_selection_when_hard_would_block():
    # Two very close candidates and one far; with hard constraint close ones would be blocked
    latlon = [(0.0, 0.0), (0.01, 0.01), (1.0, 1.0)]
    years = [1900, 1900, 1950]
    meta = make_metadata(latlon, years)

    X = np.zeros((3, 4))

    # With spatial_penalty_weight > 0, _violates_spatial_constraint returns False
    m = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.1)
    assert m._violates_spatial_constraint(1, np.array([0])) is False
