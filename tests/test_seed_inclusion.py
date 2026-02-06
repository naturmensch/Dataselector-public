import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _install_apricot_stub(monkeypatch):
    fake = types.ModuleType("apricot")

    class _FakeFacilityLocation:
        def __init__(self, n_samples=None, metric=None, random_state=None):
            self.n_samples = n_samples
            self.metric = metric
            self.random_state = random_state
            self.ranking = []

        def fit(self, X):
            n = X.shape[0]
            k = min(self.n_samples or n, n)
            self.ranking = list(range(k))

    fake.FacilityLocationSelection = _FakeFacilityLocation
    monkeypatch.setitem(sys.modules, "apricot", fake)


def test_seed_included_by_index(monkeypatch):
    _install_apricot_stub(monkeypatch)
    # Import module after stubbing so apricot is resolved during import
    import dataselector.selection.diversity_selector as ds_mod

    importlib.reload(ds_mod)
    DiversitySelector = ds_mod.DiversitySelector

    meta = pd.DataFrame(
        {
            "longName": ["SEED_001", "other"],
            "N": [50.0, 51.0],
            "left": [7.0, 8.0],
            "year": [1900, 1901],
        }
    )
    features = np.zeros((2, 64))

    ds = DiversitySelector(n_samples=2, use_multi_criteria=False)
    selected = ds.select(
        features=features, metadata=meta, spatial_constraint=False, pre_selected=[0]
    )

    assert 0 in selected, "Pre-selected seed (index) not included"
    assert len(selected) == 2


def test_seed_included_by_name(monkeypatch):
    # Provide fake MultiCriteria implementation which respects preselected indices
    fake_mod = types.ModuleType("dataselectormulti_criteria_facility_location")

    class FakeMC:
        def __init__(
            self,
            n_samples=None,
            metadata=None,
            alpha_visual=None,
            beta_spatial=None,
            gamma_temporal=None,
            min_distance_km=None,
            metric=None,
            random_state=None,
            preselected_indices=None,
        ):
            self.n_samples = n_samples
            self.metadata = metadata
            self.preselected_indices = (
                list(preselected_indices) if preselected_indices is not None else []
            )
            self.ranking = []

        def fit(self, X):
            n = X.shape[0]
            rest = [i for i in range(n) if i not in self.preselected_indices]
            ranking = list(self.preselected_indices) + rest
            self.ranking = ranking[: self.n_samples]

    setattr(fake_mod, "MultiCriteriaFacilityLocation", FakeMC)
    monkeypatch.setitem(
        sys.modules, "dataselectormulti_criteria_facility_location", fake_mod
    )

    # reload diversity_selector so it picks up the fake module
    import dataselector.selection.diversity_selector as ds_mod

    importlib.reload(ds_mod)
    DiversitySelector = ds_mod.DiversitySelector

    meta = pd.DataFrame(
        {
            "longName": ["SEED_001", "other"],
            "N": [50.0, 51.0],
            "left": [7.0, 8.0],
            "year": [1900, 1901],
        }
    )
    features = np.zeros((2, 64))

    ds = DiversitySelector(n_samples=2, use_multi_criteria=True)
    selected = ds.select(
        features=features,
        metadata=meta,
        spatial_constraint=False,
        pre_selected_names=["SEED_001"],
    )

    assert 0 in selected, "Pre-selected seed (name) not included"
    assert len(selected) == 2


def test_seed_included_by_substring_case_insensitive(monkeypatch):
    # Ensure name matching is case-insensitive and supports substring matches in longName
    fake_mod = types.ModuleType("dataselectormulti_criteria_facility_location")

    class FakeMC2:
        def __init__(
            self,
            n_samples=None,
            metadata=None,
            alpha_visual=None,
            beta_spatial=None,
            gamma_temporal=None,
            min_distance_km=None,
            metric=None,
            random_state=None,
            preselected_indices=None,
        ):
            self.n_samples = n_samples
            self.metadata = metadata
            self.preselected_indices = (
                list(preselected_indices) if preselected_indices is not None else []
            )
            self.ranking = []

        def fit(self, X):
            n = X.shape[0]
            rest = [i for i in range(n) if i not in self.preselected_indices]
            ranking = list(self.preselected_indices) + rest
            self.ranking = ranking[: self.n_samples]

    setattr(fake_mod, "MultiCriteriaFacilityLocation", FakeMC2)
    monkeypatch.setitem(
        sys.modules, "dataselectormulti_criteria_facility_location", fake_mod
    )

    import dataselector.selection.diversity_selector as ds_mod

    importlib.reload(ds_mod)
    DiversitySelector = ds_mod.DiversitySelector

    meta = pd.DataFrame(
        {
            "longName": ["KDR_001.png", "other"],
            "N": [50.0, 51.0],
            "left": [7.0, 8.0],
            "year": [1900, 1901],
        }
    )
    features = np.zeros((2, 64))

    ds = DiversitySelector(n_samples=2, use_multi_criteria=True)
    selected = ds.select(
        features=features,
        metadata=meta,
        spatial_constraint=False,
        pre_selected_names=["kdr_001"],
    )

    assert 0 in selected, "Pre-selected seed (substring/case-insensitive) not included"
    assert len(selected) == 2
