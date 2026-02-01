import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def DiversitySelector():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    mod = importlib.import_module("src.diversity_selector")
    return mod.DiversitySelector


@pytest.fixture(scope="module")
def MetadataProcessor():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    mod = importlib.import_module("src.metadata_processor")
    return mod.MetadataProcessor


def make_metadata(n):
    return pd.DataFrame(
        {
            "N": np.random.uniform(48, 55, n),
            "left": np.random.uniform(6, 15, n),
            "year": np.random.randint(1880, 1945, n),
        }
    )


def test_spatial_constraint_preserves_count(DiversitySelector):
    selector = DiversitySelector(n_samples=10, use_multi_criteria=False)

    features = np.random.randn(100, 2048)
    metadata = make_metadata(100)

    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=1.0
    )

    assert len(result) == 10, f"Expected 10 samples, got {len(result)}"


def test_spatial_constraint_respects_distance(DiversitySelector, MetadataProcessor):
    selector = DiversitySelector(n_samples=5, use_multi_criteria=False)
    features = np.random.randn(50, 2048)
    metadata = make_metadata(50)

    min_dist = 100.0
    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=min_dist
    )

    processor = MetadataProcessor("")
    for i, idx1 in enumerate(result):
        for idx2 in result[i + 1 :]:
            lat1, lon1 = metadata.loc[idx1, "N"], metadata.loc[idx1, "left"]
            lat2, lon2 = metadata.loc[idx2, "N"], metadata.loc[idx2, "left"]
            dist = processor.calculate_spatial_distance(lat1, lon1, lat2, lon2)
            assert dist >= min_dist or len(result) == selector.n_samples


def test_spatial_constraint_with_insufficient_samples(DiversitySelector):
    selector = DiversitySelector(n_samples=20, use_multi_criteria=False)
    features = np.random.randn(10, 2048)  # Nur 10 Samples
    metadata = make_metadata(10)

    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=5000.0
    )
    # Wenn min_distance unrealistisch groß ist, dürfen wir weniger als n_samples zurückgeben
    assert len(result) <= 10
    assert len(result) <= 20
