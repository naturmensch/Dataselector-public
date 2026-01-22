import numpy as np
import pytest

from src.clustering import ClusteringPipeline

pytestmark = pytest.mark.integration

@pytest.fixture(autouse=True)
def _require_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_errors_before_running():
    cl = ClusteringPipeline(n_clusters=3)
    with pytest.raises(ValueError):
        cl.get_cluster_statistics()
    with pytest.raises(ValueError):
        cl.get_cluster_centers()


def test_fit_transform_and_predict():
    rng = np.random.RandomState(0)
    features = rng.normal(size=(20, 32))
    cl = ClusteringPipeline(n_clusters=3, umap_n_components=2, random_state=0)
    emb, labels = cl.fit_transform(features)
    assert emb.shape == (20, 2)
    assert labels.shape == (20,)
    stats = cl.get_cluster_statistics()
    assert stats["total_samples"] == 20
    # predict on a subset
    new = rng.normal(size=(5, 32))
    preds = cl.predict_cluster(new)
    assert preds.shape == (5,)


def test_get_samples_per_cluster_and_distances():
    rng = np.random.RandomState(1)
    features = rng.normal(size=(30, 16))
    cl = ClusteringPipeline(n_clusters=4, umap_n_components=2, random_state=1)
    cl.fit_transform(features)
    # should return arrays without error
    arr = cl.get_samples_per_cluster(0)
    assert isinstance(arr, np.ndarray)
    dists = cl.calculate_intra_cluster_distances()
    assert isinstance(dists, dict)
    assert all(isinstance(v, float) for v in dists.values())


def test_adaptive_n_neighbors_values():
    rng = np.random.RandomState(2)
    N = 100
    features = rng.normal(size=(N, 8))
    cl = ClusteringPipeline(n_clusters=5, umap_n_components=2, random_state=2)
    emb, labels = cl.fit_transform(features)
    # Expect n_neighbors to be approx sqrt(100)=10
    assert cl.umap_reducer.n_neighbors == 10


def test_small_n_fallback():
    rng = np.random.RandomState(3)
    features = rng.normal(size=(2, 8))
    cl = ClusteringPipeline(n_clusters=2, umap_n_components=2, random_state=3)
    emb, labels = cl.fit_transform(features)
    assert emb.shape == (2, 2)
    assert (labels == 0).all()


def test_persist_n_neighbors_to_manifest(tmp_path, monkeypatch):
    # Create an ExperimentManager and set EXPERIMENT_RUN_DIR so clustering will persist config
    from src.experiment_manager import ExperimentManager

    em = ExperimentManager(
        name="test_clust_persist", description="test", base_dir=tmp_path
    )
    em.save_manifest()
    monkeypatch.setenv("EXPERIMENT_RUN_DIR", str(em.run_dir))

    rng = np.random.RandomState(4)
    N = 25
    features = rng.normal(size=(N, 8))
    cl = ClusteringPipeline(n_clusters=3, umap_n_components=2, random_state=4)
    cl.fit_transform(features)

    # Check that config file was saved into run config
    cfg_file = em.run_dir / "config" / "config_clustering.yaml"
    assert cfg_file.exists()
    import yaml

    cfg = yaml.safe_load(cfg_file.read_text())
    assert "n_neighbors" in cfg and isinstance(cfg["n_neighbors"], int)
    # Basic sanity: value should be >1 and < N
    assert 1 < cfg["n_neighbors"] < N
