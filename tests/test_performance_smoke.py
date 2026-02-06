import sys
import time
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


@pytest.mark.slow
def test_selection_completes_on_moderate_size(monkeypatch):
    """Smoke test: ensure `DiversitySelector.select` completes on moderate input size."""
    _install_apricot_stub(monkeypatch)
    import importlib

    import dataselector.selection.diversity_selector as ds_mod
    importlib.reload(ds_mod)
    DiversitySelector = ds_mod.DiversitySelector

    n = 200
    dim = 64
    features = np.random.randn(n, dim)
    meta = pd.DataFrame({"N": np.linspace(48.0, 52.0, n), "left": np.linspace(6.0, 10.0, n), "year": np.linspace(1890, 1930, n)})

    sel = DiversitySelector(n_samples=34, use_multi_criteria=False)
    t0 = time.time()
    idx = sel.select(features, metadata=meta, spatial_constraint=False)
    elapsed = time.time() - t0

    assert len(idx) == min(34, n)
    # This test is a smoke check; timing is informative and not asserted to avoid flakiness
    print(f"selection completed in {elapsed:.2f}s on {n} samples")