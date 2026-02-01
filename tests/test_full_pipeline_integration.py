import importlib.util
import sys
import os
from pathlib import Path
import subprocess
import textwrap
import time

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

<<<<<<< HEAD
# Inject a lightweight fake `src` package into sys.modules early to avoid importing
# the real package `src` (which triggers heavy imports like umap/numba during test import).
# Individual tests will override submodules as needed.
import types
if 'src' not in sys.modules:
    mod = types.ModuleType('src')
    # mark as package so submodule imports (e.g., src.cache) work during tests
    mod.__path__ = []
    sys.modules['src'] = mod
else:
    # if present but not package-like, add __path__ to allow submodule imports
    if not hasattr(sys.modules['src'], '__path__'):
        sys.modules['src'].__path__ = []

=======
>>>>>>> origin/feat/cache-by-hash

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
        "longName,shortName,N,left,image_path,image_filename,year\n"
        + "\n".join([f"A,i,{i},10,,," for i in range(4)])
    )

    # Before loading src/io.py, inject lightweight stubs for src submodules to avoid heavy imports
    import types

    fake_feat = types.ModuleType("src.feature_extractor")
    class _FE:
        def __init__(self, *a, **k):
            pass
        def extract_features_batch(self, image_paths, data_dir, batch_size=16):
            # return zeros matching number of image_paths
            return np.zeros((len(image_paths), 16), dtype=np.float32)
    fake_feat.FeatureExtractor = _FE

    fake_meta = types.ModuleType("src.metadata_processor")
    class _MP:
        def __init__(self, csv_path):
            self.csv_path = csv_path
        def load_csv(self):
            return pd.read_csv(self.csv_path)
        def add_temporal_metadata(self):
            return self.load_csv()
        def resolve_image_paths(self, image_dir):
            df = self.load_csv()
            df['image_path'] = df.get('image_path', pd.Series([None]*len(df)))
            return df
    fake_meta.MetadataProcessor = _MP

<<<<<<< HEAD
    monkeypatch.setitem(sys.modules, 'src.feature_extractor', fake_feat)
    monkeypatch.setitem(sys.modules, 'src.metadata_processor', fake_meta)

    # The io module imports from src.cache; ensure src.cache is loadable and registered
    spec = importlib.util.spec_from_file_location("src.cache", REPO_ROOT / "src" / "cache.py")
    cache_mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "src.cache", cache_mod)
    spec.loader.exec_module(cache_mod)
=======
    sys.modules['src.feature_extractor'] = fake_feat
    sys.modules['src.metadata_processor'] = fake_meta
>>>>>>> origin/feat/cache-by-hash

    # load src/io.py as isolated module (avoid package-level side effects)
    io_mod = _load_module_from_path("test_src_io", REPO_ROOT / "src" / "io.py")

    # monkeypatch the heavy extractor to a fast deterministic stub (extra safety)
    monkeypatch.setattr(io_mod, "extract_features", lambda metadata, batch_size=16: np.zeros((len(metadata), 16), dtype=np.float32))

    feats = io_mod.load_or_extract_features(out_dir=tmp_out, csv_meta=str(meta), batch_size=4, cache=True)

    assert feats.shape[0] == 4
    assert (tmp_out / "features.npy").exists()


<<<<<<< HEAD
def test_multicriteria_fit_guard_raises_on_mismatch(monkeypatch):
=======
def test_multicriteria_fit_guard_raises_on_mismatch():
>>>>>>> origin/feat/cache-by-hash
    """MultiCriteriaFacilityLocation.fit should raise when features rows != metadata rows."""

    # Inject lightweight stub for src.spatial_facility_location used by the module
    import types
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
<<<<<<< HEAD
    monkeypatch.setitem(sys.modules, 'src.spatial_facility_location', fake_spatial)
=======
    sys.modules['src.spatial_facility_location'] = fake_spatial
>>>>>>> origin/feat/cache-by-hash

    mc_mod = _load_module_from_path(
        "test_multi", REPO_ROOT / "src" / "multi_criteria_facility_location.py"
    )

    # Create metadata with 5 rows
    metadata = pd.DataFrame({"N": np.arange(5.0), "left": np.arange(5.0), "year": np.arange(1900, 1905)})

    # instantiate with n_samples small
    mc = mc_mod.MultiCriteriaFacilityLocation(n_samples=3, metadata=metadata, alpha_visual=0.7, beta_spatial=0.15, gamma_temporal=0.15)

    # Create features with mismatched rows (4 != 5)
    X = np.zeros((4, 10))

    with pytest.raises(ValueError, match=r"Feature rows .* != metadata rows .*"):
        mc.fit(X)


def test_monitor_run_hook_with_dummy_script(tmp_path):
    """Run the monitor's run_hook against a small dummy script to emulate a full run.

    The dummy script writes a small artifact in outputs/runs/dummy_run and exits 0. The test
    validates that run_hook returns success and that the artifact exists and logs were written.
    """
    monitor_mod = _load_module_from_path("test_monitor", REPO_ROOT / "scripts" / "xxl_full_run_monitor.py")

    # create a dummy script that simulates a run
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    dummy = scripts_dir / "dummy_complete.py"
    dummy.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            import os, sys, time
            root = os.getcwd()
            outdir = os.path.join(root, 'outputs', 'runs', 'dummy_run')
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, 'results.txt'), 'w') as f:
                f.write('dummy-result')
            print('DUMMY_RUN_DONE')
            sys.stdout.flush()
            time.sleep(0.1)
            sys.exit(0)
            """
        )
    )
    dummy.chmod(0o755)

    base_log_dir = tmp_path / "logs"
    base_log_dir.mkdir()
    active_log = base_log_dir / "active.log"

<<<<<<< HEAD
    # Use run_hook to execute the dummy script; ensure subprocess runs in tmp_path
    import sys
    cmd_str = f"{sys.executable} {str(dummy)}"

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = monitor_mod.run_hook(
            name="dummy",
            cmd_str=cmd_str,
            base_log_dir=base_log_dir,
            active_log=active_log,
            timeout=10,
            retries=0,
            env=os.environ.copy(),
            start_new_session=True,
            pass_dry_run=False,
        )
    finally:
        os.chdir(old_cwd)

    assert isinstance(result, dict)
    # run_hook returns a meta dict with attempts; prefer 'success' flag and last attempt exit_code
    assert result.get("success") is True
    assert result.get("attempts") and result.get("attempts")[-1].get("exit_code") == 0

    # validate the artifact was created in the tmp_path where the subprocess ran
    artifact = tmp_path / "outputs" / "runs" / "dummy_run" / "results.txt"
    assert artifact.exists()
    assert artifact.read_text() == "dummy-result"

    # active log should exist
    assert active_log.exists()

    # The run's stdout/stderr are written to base_log_dir / f"dummy_*.log"
    run_logs = list(base_log_dir.glob("dummy_*.log"))
    assert run_logs, f"No run log found in {base_log_dir}"
    runtxt = run_logs[0].read_text()
    assert "DUMMY_RUN_DONE" in runtxt
=======
    # Use run_hook to execute the dummy script
    cmd_str = f"python {str(dummy)}"

    result = monitor_mod.run_hook(
        name="dummy",
        cmd_str=cmd_str,
        base_log_dir=base_log_dir,
        active_log=active_log,
        timeout=10,
        retries=0,
        env=os.environ.copy(),
        start_new_session=True,
        pass_dry_run=False,
    )

    assert isinstance(result, dict)
    assert result.get("exit_code") == 0

    # validate the artifact was created
    artifact = Path("outputs") / "runs" / "dummy_run" / "results.txt"
    assert artifact.exists()
    assert artifact.read_text() == "dummy-result"

    # active log should exist and contain at least one DUMMY_RUN_DONE
    assert active_log.exists()
    logtxt = active_log.read_text()
    assert "DUMMY_RUN_DONE" in logtxt or "DUMMY_RUN_DONE" in (base_log_dir / "dummy" / "dummy.log").read_text()
>>>>>>> origin/feat/cache-by-hash


if __name__ == '__main__':
    # run tests locally for debugging convenience
    import pytest

    sys.exit(pytest.main([__file__]))
