import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# repo root (two levels up from tests/integration)
REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

# Prevent importing src/__init__.py (which pulls heavy native deps like umap/numba at import time)
# by registering a lightweight package module early. This lets submodules be loaded directly
# via importlib without executing package-level side-effects.
import types

if "src" not in sys.modules:
    _src_pkg = types.ModuleType("src")
    _src_pkg.__path__ = []
    sys.modules["src"] = _src_pkg
else:
    if not hasattr(sys.modules["src"], "__path__"):
        sys.modules["src"].__path__ = []


@pytest.mark.integration
def test_cache_migration_and_load(tmp_path):
    """Migration from legacy features.npy -> hashed cache and load-back should work."""
    out = tmp_path / "outputs"
    out.mkdir()
    feats = np.arange(12).reshape(3, 4)
    np.save(out / "features.npy", feats)

    csv = tmp_path / "data.csv"
    csv.write_text("id\n1\n2\n3\n")

    # Before importing the migrate script, inject a lightweight stub for `src.cache` to
    # avoid importing heavy native deps indirectly via package-level imports.
    import sys
    import types

    fake_cache = types.ModuleType("dataselectorcache")

    def compute_meta_hash(csv_path, params=None):
        return "deadbeef" * 8  # deterministic fake hash

    def create_meta_info(csv_path, params=None):
        return {"metadata_csv": str(csv_path), "params": params}

    def atomic_write_features_with_meta(out_dir, feats, meta_hash, meta_info):
        # write a simple features-{meta_hash}.npy
        import numpy as _np

        target = Path(out_dir) / f"features-{meta_hash}.npy"
        _np.save(target, feats)

    fake_cache.compute_meta_hash = compute_meta_hash
    fake_cache.create_meta_info = create_meta_info
    fake_cache.atomic_write_features_with_meta = atomic_write_features_with_meta

    sys.modules["dataselectorcache"] = fake_cache

    # Import migrate function
    spec = importlib.util.spec_from_file_location(
        "migrate_mod", REPO_ROOT / "scripts" / "migrate_feature_cache_to_hash.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    code = mod.migrate(out, csv, dry_run=False)
    assert code == 0

    assert not (out / "features.npy").exists()
    matches = list(out.glob("features-*.npy"))
    assert len(matches) == 1


@pytest.mark.integration
def test_feature_cache_validation_reextracts(tmp_path, monkeypatch):
    """load_or_extract_features should detect stale features.npy rows != metadata and re-extract."""
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

    # lazy import helpers from tests.utils to avoid heavy imports at collection
    from tests.utils import load_module_from_path

    # stub heavy feature extractor and metadata processor
    fake_feat = types.ModuleType("dataselectorfeature_extractor")

    class _FE:
        def __init__(self, *a, **k):
            pass

        def extract_features_batch(self, image_paths, data_dir, batch_size=16):
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

        def ensure_metric_crs(self, target_epsg: int = 25832):
            """Return None if no GeoDataFrame available (expected in this test)."""
            return None

    fake_meta.MetadataProcessor = _MP

    monkeypatch.setitem(sys.modules, "dataselectorfeature_extractor", fake_feat)
    monkeypatch.setitem(sys.modules, "dataselectormetadata_processor", fake_meta)

    # ensure src.cache is loadable
    spec = importlib.util.spec_from_file_location(
        "dataselectorcache", REPO_ROOT / "dataselector" / "pipeline" / "cache.py"
    )
    cache_mod = importlib.util.module_from_spec(spec)
    sys.modules["dataselectorcache"] = cache_mod
    spec.loader.exec_module(cache_mod)

    io_mod = load_module_from_path(
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
    assert (tmp_out / "features.npy").exists() or list(tmp_out.glob("features-*.npy"))


@pytest.mark.integration
def test_pipeline_smoke_small(tmp_path, monkeypatch):
    """Small smoke run through ExperimentRunner with stubs for heavy deps."""
    root = tmp_path / "repo"
    root.mkdir()
    data_dir = root / "data"
    outputs = root / "outputs"
    data_dir.mkdir()
    (data_dir / "images").mkdir()
    outputs.mkdir()

    # create tiny CSV with 4 records
    csv_meta = data_dir / "new_all_tiles.csv"
    rows = ["longName,shortName,N,left,image_path,image_filename,year"]
    for i in range(4):
        img = data_dir / "images" / f"KDR_{i:03d}.png"
        img.write_bytes(b"png")
        rows.append(
            f"KDR_{i:03d}.png,KDR_{i:03d},{50.0+i},{8.0+i},{img},{img.name},{1890+i}"
        )
    csv_meta.write_text("\n".join(rows))

    # stub FeatureExtractor and apricot
    from tests.utils import FakeFeatureExtractor, load_module_from_path

    fake_feat = types.ModuleType("dataselectorfeature_extractor")
    fake_feat.FeatureExtractor = FakeFeatureExtractor
    monkeypatch.setitem(sys.modules, "src.feature_extractor", fake_feat)

    fake_apricot = types.ModuleType("apricot")

    class _FakeFL:
        def __init__(self, n_samples=None, metric=None):
            self.n_samples = n_samples
            self.metric = metric
            self.gains_ = None
            self.ranking = None

        def fit(self, X):
            n = X.shape[0]
            self.ranking = np.arange(n)[: self.n_samples]
            self.gains_ = np.ones(n)

    fake_apricot.FacilityLocationSelection = _FakeFL
    monkeypatch.setitem(sys.modules, "apricot", fake_apricot)

    # load lightweight modules
    spatial_mod = load_module_from_path(
        "spatial_mod",
        REPO_ROOT / "dataselector" / "selection" / "spatial_facility_location.py",
    )
    monkeypatch.setitem(sys.modules, "src.spatial_facility_location", spatial_mod)

    mc_mod = load_module_from_path(
        "mc_mod",
        REPO_ROOT
        / "dataselector"
        / "selection"
        / "multi_criteria_facility_location.py",
    )
    monkeypatch.setitem(sys.modules, "src.multi_criteria_facility_location", mc_mod)

    io_mod = load_module_from_path(
        "io_mod", REPO_ROOT / "dataselector" / "data" / "io.py"
    )
    monkeypatch.setitem(sys.modules, "src.io", io_mod)

    # Provide a fake DiversitySelector to avoid importing heavy package-level deps (umap/numba)
    fake_divsel = types.ModuleType("src.diversity_selector")

    class _FakeDS:
        def __init__(
            self,
            n_samples=5,
            use_multi_criteria=False,
            use_constraint_integration=False,
            **kwargs,
        ):
            # Accept extra kwargs (random_state, etc.) to be compatible with real API
            self.n_samples = n_samples
            self.use_multi_criteria = use_multi_criteria
            self.use_constraint_integration = use_constraint_integration
            for k, v in kwargs.items():
                setattr(self, k, v)

        def select(self, features, metadata, *a, **k):
            return list(range(min(self.n_samples, len(features))))

        def export_selection(self, metadata, out_file):
            import pandas as _pd

            sel = _pd.DataFrame(
                {"selection_rank": list(range(min(self.n_samples, len(metadata))))}
            )
            sel.to_csv(out_file, index=False)
            return sel

    fake_divsel.DiversitySelector = _FakeDS
    monkeypatch.setitem(sys.modules, "src.diversity_selector", fake_divsel)

    # Provide a lightweight `src.metrics` to avoid importing heavy deps
    fake_metrics = types.ModuleType("src.metrics")

    def compute_metrics(
        selected_idx=None, metadata=None, cluster_labels=None, features=None
    ):
        # Minimal metrics consistent with src.metrics.compute_metrics
        n_selected = len(selected_idx) if selected_idx is not None else 0
        temporal_std = 0.0
        spatial_mean_km = 0.0
        clusters_covered = (
            int(len(set(cluster_labels[selected_idx])))
            if (cluster_labels is not None and selected_idx is not None)
            else 0
        )
        return {
            "n_selected": n_selected,
            "temporal_std": temporal_std,
            "spatial_mean_km": spatial_mean_km,
            "clusters_covered": clusters_covered,
        }

    fake_metrics.compute_metrics = compute_metrics
    monkeypatch.setitem(sys.modules, "src.metrics", fake_metrics)

    experiments = load_module_from_path(
        "experiments", REPO_ROOT / "dataselector" / "pipeline" / "experiments.py"
    )
    monkeypatch.setitem(sys.modules, "src.experiments", experiments)

    # monkeypatch KMeans to simple fake
    class FakeKMeans:
        def __init__(self, n_clusters=8, random_state=None):
            self.n_clusters = n_clusters
            self.random_state = random_state

        def fit_predict(self, X):
            n = X.shape[0]
            labels = np.arange(n) % max(1, self.n_clusters)
            return labels

    monkeypatch.setattr(experiments, "KMeans", FakeKMeans)

    # run io cache generation
    io_mod.load_or_extract_features(
        out_dir=str(outputs), csv_meta=str(csv_meta), batch_size=4, cache=True
    )
    found = list(outputs.glob("features-*.npy"))
    assert len(found) >= 1

    # Run a short sweep
    runner = experiments.ExperimentRunner(output_dir=str(outputs / "tuning_fast"))
    df = runner.run_weight_sweep(
        csv_meta=str(csv_meta),
        n_samples=3,
        weight_combinations=[(0.7, 0.2, 0.1), (0.6, 0.3, 0.1)],
        n_clusters=2,
        batch_size=4,
        min_distance_km=0.0,
        patience=1,
        max_runs=2,
    )

    assert (outputs / "tuning_fast" / "tuning_results.csv").exists()
    assert (outputs / "tuning_fast" / "meta.json").exists()
