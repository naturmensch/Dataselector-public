import os
import sys
import pytest

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from tests.utils import load_module_from_path


# Prefer shared fixtures to centralize data generation
# Use small dim by default to keep integration tests fast


@pytest.fixture
def make_metadata_local(make_dummy_metadata):
    return make_dummy_metadata


@pytest.fixture
def make_features_local(make_features):
    return lambda n, dim=32, seed=0: make_features(n, dim=dim, seed=seed)


def test_end_to_end_selection_and_export(tmp_path, make_features_local, make_metadata_local):
    # Try to import the real selector; skip if native deps are incompatible in this environment
    try:
        from src.diversity_selector import DiversitySelector
    except Exception as e:
        pytest.skip(f"Skipping integration test due to import error: {e}")

    n_candidates = 100
    n_select = 10
    features = make_features_local(n_candidates)
    metadata = make_metadata_local(n_candidates)

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


def test_all_three_modes_run(make_features_local, make_metadata_local):
    try:
        from src.diversity_selector import DiversitySelector
    except Exception as e:
        pytest.skip(f"Skipping integration test due to import error: {e}")

    n_candidates = 80
    features = make_features_local(n_candidates)
    metadata = make_metadata_local(n_candidates)

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


def test_coverage_statistics(make_features_local, make_metadata_local):
    try:
        from src.diversity_selector import DiversitySelector
    except Exception as e:
        pytest.skip(f"Skipping integration test due to import error: {e}")

    n = 50
    features = make_features_local(n)
    metadata = make_metadata_local(n)

    selector = DiversitySelector(n_samples=8, use_multi_criteria=False)
    selector.select(features, metadata, spatial_constraint=False)

    # fake clustering labels
    cluster_labels = np.random.randint(0, 5, size=n)
    stats = selector.get_coverage_statistics(features, cluster_labels)

    assert "n_selected" in stats and stats["n_selected"] == 8
    assert "clusters_covered" in stats
    assert isinstance(stats["cluster_distribution"], dict)
    assert stats["diversity_score"] >= 0.0
