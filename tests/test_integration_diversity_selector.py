import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", exc_type=ImportError)
pytestmark = pytest.mark.integration

from src.diversity_selector import DiversitySelector


def make_metadata(n):
    return pd.DataFrame(
        {
            "N": np.linspace(48.0, 55.0, n),
            "left": np.linspace(6.0, 15.0, n),
            "year": np.linspace(1880, 1945, n, dtype=int),
        }
    )


def make_features(n, dim=64):
    # Lower dim to speed tests
    rng = np.random.RandomState(42)
    return rng.randn(n, dim)


def test_end_to_end_selection_and_export(tmp_path):
    n_candidates = 100
    n_select = 10
    features = make_features(n_candidates)
    metadata = make_metadata(n_candidates)

    selector = DiversitySelector(n_samples=n_select, use_multi_criteria=False)
    selected = selector.select(
        features, metadata, spatial_constraint=True, min_distance_km=1.0
    )

    # check selection size
    assert len(selected) == n_select

    # export and check file
    out_file = tmp_path / "selection.csv"
    df = selector.export_selection(metadata, str(out_file))

    assert os.path.exists(str(out_file))
    assert len(df) == n_select
    assert "selection_rank" in df.columns


def test_all_three_modes_run():
    n_candidates = 80
    features = make_features(n_candidates)
    metadata = make_metadata(n_candidates)

    # Legacy mode
    sel_legacy = DiversitySelector(
        n_samples=5, use_multi_criteria=False, use_constraint_integration=False
    )
    res_legacy = sel_legacy.select(features, metadata, spatial_constraint=False)
    assert len(res_legacy) == 5

    # Constraint-integrated
    sel_constraint = DiversitySelector(
        n_samples=6, use_multi_criteria=False, use_constraint_integration=True
    )
    res_constraint = sel_constraint.select(
        features, metadata, spatial_constraint=True, min_distance_km=1.0
    )
    assert len(res_constraint) == 6

    # Multi-criteria
    sel_multi = DiversitySelector(n_samples=7, use_multi_criteria=True)
    res_multi = sel_multi.select(
        features, metadata, alpha_visual=0.6, beta_spatial=0.2, gamma_temporal=0.2
    )
    assert len(res_multi) == 7


def test_coverage_statistics():
    n = 50
    features = make_features(n)
    metadata = make_metadata(n)

    selector = DiversitySelector(n_samples=8, use_multi_criteria=False)
    selector.select(features, metadata, spatial_constraint=False)

    # fake clustering labels
    cluster_labels = np.random.randint(0, 5, size=n)
    stats = selector.get_coverage_statistics(features, cluster_labels)

    assert "n_selected" in stats and stats["n_selected"] == 8
    assert "clusters_covered" in stats
    assert isinstance(stats["cluster_distribution"], dict)
    assert stats["diversity_score"] >= 0.0
