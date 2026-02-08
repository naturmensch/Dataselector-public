from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from dataselector.tools.audit import extract_bounds_from_aux
from dataselector.workflows.bootstrap import bootstrap_candidate


def test_extract_bounds_from_aux_logs_parse_warning(tmp_path: Path, caplog):
    aux_path = tmp_path / "broken.aux.xml"
    aux_path.write_text("<GeoTransform>1,2,3", encoding="utf-8")

    caplog.set_level(logging.WARNING, logger="dataselector.tools.audit")
    result = extract_bounds_from_aux(aux_path)

    assert result is None
    assert any(
        "Failed to parse aux.xml" in rec.getMessage()
        and str(aux_path) in rec.getMessage()
        for rec in caplog.records
    )


def test_bootstrap_candidate_logs_clustering_failures(monkeypatch, caplog):
    fake_metrics = types.ModuleType("dataselector.analysis.metrics")
    fake_metrics.compute_metrics = lambda *_a, **_k: {
        "temporal_std": 1.0,
        "spatial_mean_km": 2.0,
        "wwi_percent": 3.0,
        "clusters_covered": 4.0,
        "n_selected": 2,
    }
    monkeypatch.setitem(sys.modules, "dataselector.analysis.metrics", fake_metrics)

    fake_io = types.ModuleType("dataselector.data.io")
    fake_io.attach_metric_gdf = lambda *_a, **_k: None
    fake_io.get_metric_gdf = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "dataselector.data.io", fake_io)

    fake_clustering = types.ModuleType("dataselector.selection.clustering")

    class FailingClustering:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, _feat):
            raise RuntimeError("synthetic clustering failure")

    fake_clustering.ClusteringPipeline = FailingClustering
    monkeypatch.setitem(
        sys.modules, "dataselector.selection.clustering", fake_clustering
    )

    fake_selector = types.ModuleType("dataselector.selection.diversity_selector")

    class DummySelector:
        def __init__(self, *args, **kwargs):
            pass

        def select(self, features, metadata, **kwargs):
            return np.arange(min(2, len(features)))

    fake_selector.DiversitySelector = DummySelector
    monkeypatch.setitem(
        sys.modules, "dataselector.selection.diversity_selector", fake_selector
    )

    metadata = pd.DataFrame(
        {
            "ul_x": [500000, 500100, 500200, 500300],
            "ul_y": [5900000, 5900100, 5900200, 5900300],
            "lr_x": [500050, 500150, 500250, 500350],
            "lr_y": [5899950, 5900050, 5900150, 5900250],
            "year": [1900, 1901, 1902, 1903],
        }
    )
    features = (
        np.random.default_rng(0).normal(size=(len(metadata), 4)).astype("float32")
    )
    cluster_labels_full = np.zeros(len(metadata), dtype=int)
    original_selection = [0, 1]

    caplog.set_level(logging.WARNING, logger="dataselector.workflows.bootstrap")
    df = bootstrap_candidate(
        alpha=0.6,
        beta=0.2,
        gamma=0.2,
        min_d=10.0,
        features=features,
        metadata=metadata,
        original_selection=original_selection,
        cluster_labels_full=cluster_labels_full,
        n_boot=3,
        random_seed=1,
    )

    assert len(df) == 3
    warnings = [r for r in caplog.records if "clustering failed" in r.getMessage()]
    assert len(warnings) == 3
