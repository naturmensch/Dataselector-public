import numpy as np
from src import sampling_strategies as ss


def test_sobol_unit_cube_shape():
    arr = ss.sample_unit_hypercube_sobol(10, 3, seed=123)
    assert arr.shape == (10, 3)
    assert np.all(arr >= 0) and np.all(arr <= 1)


def test_lhs_unit_cube_shape():
    arr = ss.sample_unit_hypercube_lhs(12, 4, seed=42)
    assert arr.shape == (12, 4)
    assert np.all(arr >= 0) and np.all(arr <= 1)


def test_weights_simplex_sum_to_one():
    ws = ss.sample_weights_on_simplex_sobol(20, dim=3, seed=7)
    for w in ws:
        assert abs(sum(w) - 1.0) < 1e-8

    ws_lhs = ss.sample_weights_on_simplex_lhs(15, dim=3, seed=7)
    for w in ws_lhs:
        assert abs(sum(w) - 1.0) < 1e-8
