import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Inject a lightweight fake `src` package into sys.modules early to avoid importing
# the real package `src` (which triggers heavy imports like umap/numba during test import).
# Individual tests will override submodules as needed.
import types

if "src" not in sys.modules:
    mod = types.ModuleType("src")
    # mark as package so submodule imports (e.g., src.cache) work during tests
    mod.__path__ = []
    sys.modules["src"] = mod
else:
    # if present but not package-like, add __path__ to allow submodule imports
    if not hasattr(sys.modules["src"], "__path__"):
        sys.modules["src"].__path__ = []


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_feature_cache_validation(tmp_path, monkeypatch):
    """Ensure stale outputs/features.npy is detected and re-extracted."""
    tmp_out = tmp_path / "outputs"
    tmp_out.mkdir()

    # create a stale features.npy with shape (2,16)
    np.save(tmp_out / "features.npy", np.zeros((2, 16), dtype=np.float32))

    # create a metadata.csv with 4 rows
    meta = tmp_out / "metadata.csv"
    meta.write_text(
        "longName,shortName,ul_x,ul_y,lr_x,lr_y,image_path,image_filename,year\n"
        + "\n".join(
            [
                f"A,i,{9.95+i*0.1},{50.05+i*0.1},{10.05+i*0.1},{49.95+i*0.1},,,"
                for i in range(4)
            ]
        )
    )

    # Before loading src/io.py, inject lightweight stubs for src submodules to avoid heavy imports
    import types

    fake_feat = types.ModuleType("dataselectorfeature_extractor")

    class _FE:
        def __init__(self, *a, **k):
            pass

        def extract_features_batch(self, image_paths, data_dir, batch_size=16):
            # return zeros matching number of image_paths
            return np.zeros((len(image_paths), 16), dtype=np.float32)

    fake_feat.FeatureExtractor = _FE

    fake_meta = types.ModuleType("dataselectormetadata_processor")

    class _MP:
        def __init__(self, csv_path):
            self.csv_path = csv_path

        def load_csv(self):
            return pd.read_csv(self.csv_path)

        def add_temporal_metadata(self):
            return self.load_csv()

        def resolve_image_paths(self, image_dir):
            df = self.load_csv()
            df["image_path"] = df.get("image_path", pd.Series([None] * len(df)))
            return df

    fake_meta.MetadataProcessor = _MP

    monkeypatch.setitem(sys.modules, "src.feature_extractor", fake_feat)
    monkeypatch.setitem(sys.modules, "src.metadata_processor", fake_meta)

    # The io module imports from src.cache; ensure src.cache is loadable and registered
    spec = importlib.util.spec_from_file_location(
        "dataselectorcache", REPO_ROOT / "dataselector" / "pipeline" / "cache.py"
    )
    cache_mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dataselectorcache", cache_mod)
    spec.loader.exec_module(cache_mod)

    # load src/io.py as isolated module (avoid package-level side effects)
    io_mod = _load_module_from_path(
        "test_src_io", REPO_ROOT / "dataselector" / "data" / "io.py"
    )

    # monkeypatch the heavy extractor to a fast deterministic stub (extra safety)
    monkeypatch.setattr(
        io_mod,
        "extract_features",
        lambda metadata, batch_size=16: np.zeros((len(metadata), 16), dtype=np.float32),
    )

    feats = io_mod.load_or_extract_features(
        out_dir=tmp_out, csv_meta=str(meta), batch_size=4, cache=True
    )

    assert feats.shape[0] == 4
    assert (tmp_out / "features.npy").exists()


def test_multicriteria_fit_guard_raises_on_mismatch(monkeypatch):
    """MultiCriteriaFacilityLocation.fit should raise when features rows != metadata rows."""

    # Inject lightweight stub for src.spatial_facility_location used by the module
    import types

    fake_spatial = types.ModuleType("dataselectorspatial_facility_location")

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

    mc_mod = _load_module_from_path(
        "test_multi",
        REPO_ROOT
        / "dataselector"
        / "selection"
        / "multi_criteria_facility_location.py",
    )

    # Create metadata with 5 rows
    metadata = pd.DataFrame(
        {
            "ul_x": np.arange(5.0) - 0.05,
            "ul_y": np.arange(5.0) + 0.05,
            "lr_x": np.arange(5.0) + 0.05,
            "lr_y": np.arange(5.0) - 0.05,
            "year": np.arange(1900, 1905),
        }
    )

    # instantiate with n_samples small
    mc = mc_mod.MultiCriteriaFacilityLocation(
        n_samples=3,
        metadata=metadata,
        alpha_visual=0.7,
        beta_spatial=0.15,
        gamma_temporal=0.15,
    )

    # Create features with mismatched rows (4 != 5)
    X = np.zeros((4, 10))

    with pytest.raises(ValueError, match=r"Feature rows .* != metadata rows .*"):
        mc.fit(X)


def test_xxl_monitor_delegates_to_cli(monkeypatch):
    """Package monitor wrapper should delegate to canonical CLI command."""
    from dataselector.workflows import xxl_monitor

    called = {}

    class DummyProc:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(cmd):
        called["cmd"] = cmd
        return DummyProc(returncode=0)

    monkeypatch.setattr(xxl_monitor.subprocess, "run", fake_run)
    rc = xxl_monitor.main(["--help"])

    assert rc == 0
    assert called["cmd"][:4] == [sys.executable, "-m", "dataselector", "xxl"]
    assert called["cmd"][-1] == "--help"


if __name__ == "__main__":
    # run tests locally for debugging convenience
    import pytest

    sys.exit(pytest.main([__file__]))
