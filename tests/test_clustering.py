import numpy as np
import pytest

from dataselector.selection.clustering import ClusteringPipeline


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
