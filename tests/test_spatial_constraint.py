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


@pytest.fixture(scope="module")
def MetadataProcessor():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    mod = importlib.import_module("dataselector.data.metadata_processor")
    return mod.MetadataProcessor


def make_metadata(n):
    center_x = np.random.uniform(400000, 700000, n)
    center_y = np.random.uniform(5600000, 6100000, n)
    half = np.full(n, 50.0)
    return pd.DataFrame(
        {
            "ul_x": center_x - half,
            "ul_y": center_y + half,
            "lr_x": center_x + half,
            "lr_y": center_y - half,
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
    features = np.random.randn(10, 2048)
    # Deterministic metadata with large spacing (projected coordinates in meters)
    # to guarantee feasible strict hard-cut behavior.
    center_x = np.array([400000.0, 600000.0, 800000.0, 1000000.0, 1200000.0] * 2)
    center_y = np.array([5600000.0] * 10)
    half = np.full(10, 50.0)
    metadata = pd.DataFrame(
        {
            "ul_x": center_x - half,
            "ul_y": center_y + half,
            "lr_x": center_x + half,
            "lr_y": center_y - half,
            "year": np.random.randint(1880, 1945, 10),
        }
    )

    min_dist = 100.0
    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=min_dist
    )
    # Hard-cut contract: never violate constraints to fill up to requested count.
    assert 0 < len(result) <= selector.n_samples

    processor = MetadataProcessor("")
    for i, idx1 in enumerate(result):
        for idx2 in result[i + 1 :]:
            y1 = (metadata.loc[idx1, "ul_y"] + metadata.loc[idx1, "lr_y"]) / 2
            x1 = (metadata.loc[idx1, "ul_x"] + metadata.loc[idx1, "lr_x"]) / 2
            y2 = (metadata.loc[idx2, "ul_y"] + metadata.loc[idx2, "lr_y"]) / 2
            x2 = (metadata.loc[idx2, "ul_x"] + metadata.loc[idx2, "lr_x"]) / 2
            dist = processor.calculate_spatial_distance(y1, x1, y2, x2)
            assert dist >= min_dist


def test_spatial_constraint_with_insufficient_samples(DiversitySelector):
    selector = DiversitySelector(n_samples=20, use_multi_criteria=False)
    features = np.random.randn(10, 2048)  # Nur 10 Samples
    metadata = make_metadata(10)

    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=5000.0
    )
    # Hard-cut: bei unrealistisch striktem Abstand wird nicht aufgefüllt.
    assert len(result) <= 10
    assert len(result) < selector.n_samples


def test_spatial_constraint_no_silent_bypass_when_infeasible(DiversitySelector):
    selector = DiversitySelector(n_samples=5, use_multi_criteria=False)
    features = np.random.randn(8, 2048)
    # Sehr dichte Punkte (~1 km), damit 50 km Constraint unmöglich ist.
    center_x = np.linspace(500000.0, 507000.0, 8)
    center_y = np.full(8, 5800000.0)
    half = np.full(8, 50.0)
    metadata = pd.DataFrame(
        {
            "ul_x": center_x - half,
            "ul_y": center_y + half,
            "lr_x": center_x + half,
            "lr_y": center_y - half,
            "year": np.random.randint(1880, 1945, 8),
        }
    )

    result = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=50.0
    )
    # Strict hard-cut behavior: never fill to requested count by violating constraint.
    assert len(result) < selector.n_samples
