import numpy as np

from src.diversity_selector import DiversitySelector


def test_lazy_greedy_basic():
    n_candidates = 200
    n_select = 15
    rng = np.random.RandomState(0)
    features = rng.randn(n_candidates, 128)
    metadata = None

    sel = DiversitySelector(
        n_samples=n_select, use_multi_criteria=False, use_lazy_greedy=True
    )
    res = sel.select(features, metadata, spatial_constraint=False)

    assert len(res) == n_select


def test_lazy_greedy_matches_standard():
    n_candidates = 150
    n_select = 12
    rng = np.random.RandomState(1)
    features = rng.randn(n_candidates, 128)
    metadata = None

    sel_lazy = DiversitySelector(
        n_samples=n_select, use_multi_criteria=False, use_lazy_greedy=True
    )
    res_lazy = sel_lazy.select(features, metadata, spatial_constraint=False)

    sel_std = DiversitySelector(
        n_samples=n_select, use_multi_criteria=False, use_lazy_greedy=False
    )
    res_std = sel_std.select(features, metadata, spatial_constraint=False)

    # Compare diversity scores (should be equal or very close)
    from src.diversity_selector import DiversitySelector as DS

    ds = DS(n_samples=n_select)
    score_lazy = ds._calculate_diversity_score(features[res_lazy])
    score_std = ds._calculate_diversity_score(features[res_std])

    assert abs(score_lazy - score_std) < 1e-6
