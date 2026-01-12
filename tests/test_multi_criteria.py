import numpy as np
import pandas as pd
import pytest

from src.multi_criteria_facility_location import MultiCriteriaFacilityLocation


def test_multi_criteria_weight_validation():
    # Create minimal metadata
    meta = pd.DataFrame({"N": [0.0, 1.0], "left": [0.0, 1.0], "year": [1900, 1901]})

    # Weights do not sum to 1.0 -> ValueError
    with pytest.raises(ValueError):
        MultiCriteriaFacilityLocation(
            n_samples=1,
            metadata=meta,
            alpha_visual=0.5,
            beta_spatial=0.5,
            gamma_temporal=0.2,
        )


def test_greedy_selection_spatial_constraint():
    # Three points: 0 and 1 are at same coordinates (distance 0), 2 is far away
    meta = pd.DataFrame(
        {
            "N": [0.0, 0.0, 50.0],
            "left": [0.0, 0.0, 50.0],
            "year": [1900, 1900, 1950],
        }
    )

    # Create simple features so greedy selection has deterministic behavior
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    m = MultiCriteriaFacilityLocation(
        n_samples=2,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.15,
        gamma_temporal=0.15,
        min_distance_km=1.0,
    )

    # Compute pairwise distances and run greedy selection
    distances = m._compute_pairwise_distances(X)
    selected = m._greedy_selection(distances)

    # Expect that exactly one of points 0 and 1 is selected, and point 2 is selected
    assert len(selected) == 2
    assert 2 in selected
    assert (0 in selected) ^ (1 in selected)
