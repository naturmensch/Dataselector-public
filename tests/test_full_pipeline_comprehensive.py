import sys
import os
import shutil
import subprocess
import time
from pathlib import Path
import importlib.util
import types

import numpy as np
import pandas as pd
import pytest

from tests.utils import load_module_from_path, FakeFeatureExtractor, FakeMetadataProcessor, create_dummy_script

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_full_pipeline_simulation(tmp_dirs, repo_root, inject_src_stub, monkeypatch):
    """Comprehensive fast simulation of the full pipeline.

    This test creates a tiny dataset, stubs heavy components (feature extraction,
    UMAP/KMeans, expensive computations), and runs the weight-sweep and a
    short Optuna run to validate checkpointing and reporting. The goal is to
    exercise all code paths quickly while keeping fidelity to real calls.
    """

    # Use provided tmp_dirs fixture for standard layout
    root = tmp_dirs["root"]
    data_dir = tmp_dirs["data"]
    outputs = tmp_dirs["outputs"]

    # Ensure cwd is the tmp project root for subprocesses and file writes
    monkeypatch.chdir(root)

    # 1) Prepare small dataset (6 candidates)
    image_dir = data_dir / "images"

    # CSV with 6 records
    csv_meta = data_dir / "new_all_tiles.csv"
    rows = ["longName,shortName,N,left,image_path,image_filename,year"]
    for i in range(6):
        # create tiny placeholder PNG
        img = image_dir / f"KDR_{i:03d}.png"
        # small binary PNG data (1x1 pixel)
        img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c``\x00\x00\x00\x05\x00\x01\x0d\n\x2dB\x00\x00\x00\x00IEND\xaeB`\x82")
        rows.append(f"KDR_{i:03d}.png,KDR_{i:03d},{50.0+i},{8.0+i},{img},{img.name},{1890+i}")
    csv_meta.write_text("\n".join(rows))

    # 2) Create a stale features.npy that does not match metadata (simulate previous runs)
    np.save(outputs / "features.npy", np.zeros((4, 64), dtype=np.float32))  # stale: 4 rows vs 6

    # 3) Inject light-weight stubs for heavy src submodules before importing code
    # - FeatureExtractor: use shared FakeFeatureExtractor helper
    fake_feat = types.ModuleType("dataselectorfeature_extractor")
    fake_feat.FeatureExtractor = FakeFeatureExtractor
    monkeypatch.setitem(sys.modules, 'src.feature_extractor', fake_feat)

    # - MetadataProcessor: load the real implementation but register under src namespace
    mp_mod = load_module_from_path("mp_mod", repo_root / "dataselector" / "data" / "metadata_processor.py")
    fake_meta = types.ModuleType("dataselectormetadata_processor")
    fake_meta.MetadataProcessor = mp_mod.MetadataProcessor
    monkeypatch.setitem(sys.modules, 'src.metadata_processor', fake_meta)

    # - Provide a fake 'apricot' module to avoid requiring the external dependency in tests
    fake_apricot = types.ModuleType('apricot')
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
    monkeypatch.setitem(sys.modules, 'apricot', fake_apricot)

    # Load and register spatial and multi-criteria modules under src.* names
    spatial_mod = load_module_from_path('spatial_mod', repo_root / 'dataselector' / 'selection' / 'spatial_facility_location.py')
    monkeypatch.setitem(sys.modules, 'src.spatial_facility_location', spatial_mod)
    setattr(inject_src_stub, 'spatial_facility_location', spatial_mod)

    mc_mod = load_module_from_path('mc_mod', repo_root / 'dataselector' / 'selection' / 'multi_criteria_facility_location.py')
    monkeypatch.setitem(sys.modules, 'src.multi_criteria_facility_location', mc_mod)
    setattr(inject_src_stub, 'multi_criteria_facility_location', mc_mod)

    # Now load diversity_selector, io and metrics
    divsel_mod = load_module_from_path("diversity", repo_root / "dataselector" / "selection" / "diversity_selector.py")
    io_mod = load_module_from_path("io_mod", repo_root / "dataselector" / "data" / "io.py")
    metrics_mod = load_module_from_path("metrics_mod", repo_root / "dataselector" / "analysis" / "metrics.py")

    # Register under package-like names so 'from src.xxx import ...' works
    monkeypatch.setitem(sys.modules, 'src.diversity_selector', divsel_mod)
    monkeypatch.setitem(sys.modules, 'src.io', io_mod)
    monkeypatch.setitem(sys.modules, 'src.metrics', metrics_mod)

    # 6) Load the ExperimentRunner module directly (now submodules are available)
    experiments = load_module_from_path("experiments", repo_root / "dataselector" / "pipeline" / "experiments.py")
    monkeypatch.setitem(sys.modules, 'src.experiments', experiments)

    # 6) Monkeypatch heavy KMeans inside experiments to a light fake implementation
    class FakeKMeans:
        def __init__(self, n_clusters=8, random_state=None):
            self.n_clusters = n_clusters
            self.random_state = random_state

        def fit_predict(self, X):
            # simple cluster assignment: index mod n_clusters
            n = X.shape[0]
            labels = np.arange(n) % max(1, self.n_clusters)
            return labels

    monkeypatch.setattr(experiments, "KMeans", FakeKMeans)

    # 7) Monkeypatch diversity score calculation (avoid scipy dependency)
    monkeypatch.setattr(divsel_mod.DiversitySelector, "_calculate_diversity_score", lambda self, feats: float(np.mean(feats)) if len(feats)>0 else 0.0)

    # 7) Ensure cache is (re)generated via load_or_extract_features to test cache validation logic
    cache_mod = load_module_from_path('cache_mod', repo_root / 'dataselector' / 'pipeline' / 'cache.py')
    monkeypatch.setitem(sys.modules, 'src.cache', cache_mod)
    setattr(inject_src_stub, 'cache', cache_mod)

    io_mod.load_or_extract_features(out_dir=str(outputs), csv_meta=str(csv_meta), batch_size=4, cache=True)
    # check for either legacy outputs/features.npy or new hashed cache files
    found = list(outputs.glob('features-*.npy'))
    assert len(found) >= 1
    # validate shape of first cache
    _arr = np.load(found[0])
    assert _arr.shape[0] == 6

    # 8) Run the sweep with small params to exercise many branches (disable spatial constraint by min_distance_km=0)
    runner = experiments.ExperimentRunner(output_dir=str(outputs / "tuning_fast"))

    df = runner.run_weight_sweep(
        csv_meta=str(csv_meta),
        n_samples=3,
        weight_combinations=[(0.7,0.2,0.1),(0.6,0.3,0.1),(0.5,0.25,0.25)],
        n_clusters=3,
        batch_size=4,
        min_distance_km=0.0,
        patience=2,
        max_runs=3,
    )

    # Validate outputs
    assert (outputs / "tuning_fast" / "tuning_results.csv").exists()
    assert (outputs / "tuning_fast" / "meta.json").exists()

    # Legacy outputs/features.npy may be left stale; we validated the new hashed cache earlier.
    # Also verify that selection CSVs were written for each combination
    sel_files = list((outputs / "tuning_fast").glob("selection_*.csv"))
    assert len(sel_files) >= 1

    # 8) Now test Optuna checkpointing quickly by running a small Optuna run
    opt_mod = load_module_from_path("optuna_opt", repo_root / "scripts" / "optuna_optimize.py")

    # run small optimization (n_trials=4) with checkpoint_every=2
    opt_mod.run_optuna(n_trials=4, n_candidates=6, dim=8, n_samples=3, seed=123, sampler_name='tpe', exp_name='smoke_opt', checkpoint_every=2)

    # expect checkpoint files created
    checkpoints = list((root / "outputs").glob("optuna_results_checkpoint_*.csv"))
    assert len(checkpoints) >= 1

# rely on fixtures/monkeypatch/inject_src_stub to restore sys.modules and cwd


def test_corrupt_metadata_fails_fast(tmp_dirs, repo_root, monkeypatch):
    """Corrupt/invalid metadata CSV should raise a clear error early."""
    root = tmp_dirs["root"]
    data_dir = tmp_dirs["data"]
    monkeypatch.chdir(root)

    csv_meta = data_dir / "new_all_tiles.csv"
    # write a CSV missing required columns (no N/left)
    csv_meta.write_text("longName,shortName,image_path,year\nA,a,,1910\nB,b,,1912\n")

    # Load metadata processor directly and assert it raises on missing columns
    mp = load_module_from_path("mp_mod", repo_root / "dataselector" / "data" / "metadata_processor.py")
    with pytest.raises(ValueError, match=r"Fehlende Spalten in Metadaten"):
        mp.MetadataProcessor(str(csv_meta)).load_csv()


def test_feature_cache_write_permission_error(tmp_dirs, repo_root, monkeypatch):
    """Simulate PermissionError when writing cache to ensure it's handled/presented clearly."""
    root = tmp_dirs["root"]
    outputs = tmp_dirs["outputs"]
    data_dir = tmp_dirs["data"]
    monkeypatch.chdir(root)

    csv_meta = data_dir / "new_all_tiles.csv"
    csv_meta.write_text("longName,shortName,N,left,image_path,image_filename,year\nA,a,50,10,,imageA,1910\nB,b,51,11,,imageB,1912\n")

    # Load io module and monkeypatch np.save to raise PermissionError
    io_mod = load_module_from_path("io_mod", repo_root / "dataselector" / "data" / "io.py")
    monkeypatch.setattr("numpy.save", lambda *a, **k: (_ for _ in ()).throw(PermissionError("Disk full (simulated)")))

    with pytest.raises(PermissionError, match=r"Disk full \(simulated\)"):
        io_mod.load_or_extract_features(out_dir=str(outputs), csv_meta=str(csv_meta), batch_size=1, cache=True)
