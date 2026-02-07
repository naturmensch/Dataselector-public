import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def DiversitySelector():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    mod = importlib.import_module("dataselector.selection.diversity_selector")
    return mod.DiversitySelector


def make_grid_metadata(n, lon_start=6.0, lat=50.0, lon_step=0.28):
    # Produces n points along longitude with lon_step degrees (~30km at mid-latitudes)
    lons = [lon_start + i * lon_step for i in range(n)]
    lats = [lat for _ in range(n)]
    ul_x = np.array(lons) - 0.05
    ul_y = np.array(lats) + 0.05
    lr_x = np.array(lons) + 0.05
    lr_y = np.array(lats) - 0.05
    return pd.DataFrame(
        {
            "ul_x": ul_x,
            "ul_y": ul_y,
            "lr_x": lr_x,
            "lr_y": lr_y,
            "year": np.random.randint(1880, 1945, n),
        }
    )


def test_adaptive_min_distance_reaches_n_samples(DiversitySelector):
    selector = DiversitySelector(n_samples=5, use_multi_criteria=False)
    features = np.random.randn(10, 2048)
    metadata = make_grid_metadata(10)

    # Start with an initially too large min_distance that would prevent selection
    result = selector.select(
        features,
        metadata,
        spatial_constraint=True,
        min_distance_km=50.0,
        adaptive_min_distance=True,
        adaptive_step_km=5.0,
        adaptive_min_allowed_km=20.0,
    )

    assert (
        len(result) == 5
    ), f"Adaptive fallback failed to reach 5 samples, got {len(result)}"


def test_adaptive_fallback_allows_duplicates(DiversitySelector):
    # Zwei identische Punkte, min_distance zu groß, adaptive fallback auf 0
    coords = [
        (13.405, 52.52),
        (13.405, 52.52),
    ]
    features = np.zeros((2, 1))
    meta = pd.DataFrame(coords, columns=["center_x", "center_y"])
    meta["ul_x"] = meta["center_x"] - 0.05
    meta["ul_y"] = meta["center_y"] + 0.05
    meta["lr_x"] = meta["center_x"] + 0.05
    meta["lr_y"] = meta["center_y"] - 0.05
    meta["year"] = 2000
    sel = DiversitySelector(n_samples=2, use_multi_criteria=False)
    idx = sel.select(
        features,
        meta,
        spatial_constraint=True,
        min_distance_km=100,
        adaptive_min_distance=True,
        adaptive_min_allowed_km=0,
    )
    # adaptive fallback sollte beide zulassen
    assert len(idx) == 2
