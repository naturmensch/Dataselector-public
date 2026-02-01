import pandas as pd
import pytest

from src.pipeline_utils import (
    compute_adaptive_n_initial,
    compute_bootstrap_candidates,
    compute_fine_search_bounds,
    compute_optuna_bounds,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_compute_fine_search_bounds(tmp_path):
    df = pd.DataFrame({"min_distance_km": [30, 35, 40]})
    p = tmp_path / "pareto.csv"
    df.to_csv(p, index=False)
    vals = compute_fine_search_bounds(str(p))
    assert len(vals) == 5
    assert min(vals) >= 10


def test_compute_optuna_bounds(tmp_path):
    df = pd.DataFrame({"min_distance_km": [35, 40, 45]})
    p = tmp_path / "pareto.csv"
    df.to_csv(p, index=False)
    lo, hi = compute_optuna_bounds(str(p), pct=0.2)
    assert lo < hi
    assert lo >= 10


def test_compute_bootstrap_candidates(tmp_path):
    df = pd.DataFrame({"user_attrs_min_distance_km": [30, 40, 45], "value": [1, 2, 3]})
    p = tmp_path / "optuna.csv"
    df.to_csv(p, index=False)
    candidates = compute_bootstrap_candidates(str(p), delta=5)
    assert len(candidates) == 3


def test_compute_adaptive_n_initial_legacy():
    # legacy uses sqrt(n_tiles)
    assert compute_adaptive_n_initial(3, n_tiles=1000, strategy="legacy") == max(
        27, int(1000**0.5)
    )
    # when n_tiles missing, fallback to 27
    assert compute_adaptive_n_initial(3, n_tiles=None, strategy="legacy") == 27


def test_compute_adaptive_n_initial_modern():
    # for 3 dims modern rule returns at least 20
    assert compute_adaptive_n_initial(3, n_tiles=None, strategy="modern") == 20
    # for higher dims returns 2*D^2
    assert compute_adaptive_n_initial(5, n_tiles=None, strategy="modern") == 2 * 5 * 5
