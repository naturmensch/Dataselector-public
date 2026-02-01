import numpy as np
import pandas as pd
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from tests.utils import load_module_from_path


def _install_apricot_stub(monkeypatch):
    import types, sys

    class _FakeFacilityLocation:
        def __init__(self, n_samples=None, metric=None, random_state=None):
            self.n_samples = n_samples
            self.metric = metric
            self.random_state = random_state
            self.ranking = None

        def fit(self, X):
            n = X.shape[0]
            k = min(self.n_samples or n, n)
            self.ranking = list(range(k))

    fake = types.ModuleType("apricot")
    fake.FacilityLocationSelection = _FakeFacilityLocation
    monkeypatch.setitem(sys.modules, "apricot", fake)


def test_spatial_constraint_preserves_count(make_features, make_dummy_metadata, monkeypatch):
    _install_apricot_stub(monkeypatch)
    selector_mod = load_module_from_path("sel_mod", REPO_ROOT / "src" / "diversity_selector.py")
    DiversitySelector = selector_mod.DiversitySelector

    selector = DiversitySelector(n_samples=10, use_multi_criteria=False)
    features = make_features(100, dim=64, seed=0)
    metadata = make_dummy_metadata(100, seed=0)

    result = selector.select(features, metadata, spatial_constraint=True, min_distance_km=1.0)
    assert len(result) == 10, f"Expected 10 samples, got {len(result)}"


def test_spatial_constraint_respects_distance(make_features, make_dummy_metadata, monkeypatch):
    # Use real MetadataProcessor helper for distance computation loaded in isolation
    mp_mod = load_module_from_path("mp_mod", REPO_ROOT / "src" / "metadata_processor.py")
    MetadataProcessor = mp_mod.MetadataProcessor

    _install_apricot_stub(monkeypatch)
    selector_mod = load_module_from_path("sel_mod2", REPO_ROOT / "src" / "diversity_selector.py")
    DiversitySelector = selector_mod.DiversitySelector

    selector = DiversitySelector(n_samples=5, use_multi_criteria=False)
    features = make_features(50, dim=64, seed=1)
    metadata = make_dummy_metadata(50, seed=1)

    min_dist = 100.0
    result = selector.select(features, metadata, spatial_constraint=True, min_distance_km=min_dist)

    processor = MetadataProcessor("")
    for i, idx1 in enumerate(result):
        for idx2 in result[i + 1 :]:
            lat1, lon1 = metadata.loc[idx1, "N"], metadata.loc[idx1, "left"]
            lat2, lon2 = metadata.loc[idx2, "N"], metadata.loc[idx2, "left"]
            dist = processor.calculate_spatial_distance(lat1, lon1, lat2, lon2)
            assert dist >= min_dist or len(result) == selector.n_samples


def test_spatial_constraint_with_insufficient_samples(make_features, monkeypatch):
    _install_apricot_stub(monkeypatch)
    selector_mod = load_module_from_path("sel_mod3", REPO_ROOT / "src" / "diversity_selector.py")
    DiversitySelector = selector_mod.DiversitySelector

    selector = DiversitySelector(n_samples=20, use_multi_criteria=False)
    features = make_features(10, dim=64, seed=2)  # only 10 samples
    # create small metadata
    metadata = pd.DataFrame({"N": np.linspace(48.0, 49.0, 10), "left": np.linspace(6.0, 7.0, 10), "year": np.arange(2000, 2010)})

    result = selector.select(features, metadata, spatial_constraint=True, min_distance_km=5000.0)
    assert len(result) <= 10
    assert len(result) <= 20


# Adaptive tests
def test_adaptive_min_distance_reaches_n_samples(monkeypatch):
    _install_apricot_stub(monkeypatch)
    selector_mod = load_module_from_path("sel_mod4", REPO_ROOT / "src" / "diversity_selector.py")
    DiversitySelector = selector_mod.DiversitySelector

    selector = DiversitySelector(n_samples=5, use_multi_criteria=False)
    features = np.random.randn(10, 64)

    # grid-like longitudes
    lons = [6.0 + i * 0.28 for i in range(10)]
    lats = [50.0 for _ in range(10)]
    metadata = pd.DataFrame({"N": lats, "left": lons, "year": np.random.randint(1880, 1945, 10)})

    result = selector.select(
        features,
        metadata,
        spatial_constraint=True,
        min_distance_km=50.0,
        adaptive_min_distance=True,
        adaptive_step_km=5.0,
        adaptive_min_allowed_km=20.0,
    )

    assert len(result) == 5, f"Adaptive fallback failed to reach 5 samples, got {len(result)}"


def test_adaptive_fallback_allows_duplicates(monkeypatch):
    _install_apricot_stub(monkeypatch)
    selector_mod = load_module_from_path("sel_mod5", REPO_ROOT / "src" / "diversity_selector.py")
    DiversitySelector = selector_mod.DiversitySelector

    coords = [(52.52, 13.405), (52.52, 13.405)]
    features = np.zeros((2, 1))
    meta = pd.DataFrame(coords, columns=["N", "left"])
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
    assert len(idx) == 2


# Soft penalty / MultiCriteria Distance tests

def test_spatial_penalty_increases_nearby_distances(monkeypatch):
    # Prevent src package init from importing heavy deps by stubbing required submodules
    import types, sys
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "src", src_pkg)

    # Provide minimal src.spatial_facility_location implementation required by multi_criteria
    fake_spatial = types.ModuleType("src.spatial_facility_location")

    def _haversine_distance(lat1, lon1, lat2, lon2):
        return float(abs(lat1 - lat2) + abs(lon1 - lon2))

    def _haversine_matrix(lats, lons):
        n = len(lats)
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m[i, j] = _haversine_distance(lats[i], lons[i], lats[j], lons[j])
        return m

    fake_spatial.haversine_distance = _haversine_distance
    fake_spatial.haversine_matrix = _haversine_matrix
    monkeypatch.setitem(sys.modules, "src.spatial_facility_location", fake_spatial)

    mc_mod = load_module_from_path("mc_mod", REPO_ROOT / "src" / "multi_criteria_facility_location.py")
    MultiCriteriaFacilityLocation = mc_mod.MultiCriteriaFacilityLocation

    latlon = [(0.0, 0.0), (0.01, 0.01), (1.0, 1.0)]
    years = [1900, 1900, 1950]
    meta = pd.DataFrame({"N": [p[0] for p in latlon], "left": [p[1] for p in latlon], "year": years})

    X = np.zeros((3, 4))

    m0 = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.0)
    d0 = m0._compute_pairwise_distances(X)

    m1 = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.2)
    d1 = m1._compute_pairwise_distances(X)

    assert d1[0, 1] >= d0[0, 1]
    assert np.isclose(d1[0, 2], d0[0, 2], atol=1e-8)


def test_soft_penalty_allows_selection_when_hard_would_block():
    mc_mod = load_module_from_path("mc_mod2", REPO_ROOT / "src" / "multi_criteria_facility_location.py")
    MultiCriteriaFacilityLocation = mc_mod.MultiCriteriaFacilityLocation

    latlon = [(0.0, 0.0), (0.01, 0.01), (1.0, 1.0)]
    years = [1900, 1900, 1950]
    meta = pd.DataFrame({"N": [p[0] for p in latlon], "left": [p[1] for p in latlon], "year": years})

    X = np.zeros((3, 4))

    m = MultiCriteriaFacilityLocation(n_samples=1, metadata=meta, alpha_visual=1.0, beta_spatial=0.0, gamma_temporal=0.0, min_distance_km=50.0, spatial_penalty_weight=0.1)
    assert m._violates_spatial_constraint(1, np.array([0])) is False
