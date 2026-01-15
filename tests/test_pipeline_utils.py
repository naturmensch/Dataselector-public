import pandas as pd
import tempfile
from src.pipeline_utils import compute_fine_search_bounds, compute_optuna_bounds, compute_bootstrap_candidates


def test_compute_fine_search_bounds(tmp_path):
    df = pd.DataFrame({'min_distance_km': [30, 35, 40]})
    p = tmp_path / 'pareto.csv'
    df.to_csv(p, index=False)
    vals = compute_fine_search_bounds(str(p))
    assert len(vals) == 5
    assert min(vals) >= 10


def test_compute_optuna_bounds(tmp_path):
    df = pd.DataFrame({'min_distance_km': [35, 40, 45]})
    p = tmp_path / 'pareto.csv'
    df.to_csv(p, index=False)
    lo, hi = compute_optuna_bounds(str(p), pct=0.2)
    assert lo < hi
    assert lo >= 10


def test_compute_bootstrap_candidates(tmp_path):
    df = pd.DataFrame({'user_attrs_min_distance_km': [30, 40, 45], 'value': [1,2,3]})
    p = tmp_path / 'optuna.csv'
    df.to_csv(p, index=False)
    candidates = compute_bootstrap_candidates(str(p), delta=5)
    assert len(candidates) == 3
