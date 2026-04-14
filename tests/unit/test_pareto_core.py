from __future__ import annotations

import pandas as pd

from dataselector.selection.pareto import compute_pareto_front, is_dominated


def test_is_dominated_with_mixed_objectives() -> None:
    objectives = [("score", "maximize"), ("cost", "minimize")]
    a = {"score": 0.8, "cost": 10.0}
    b = {"score": 0.9, "cost": 9.0}

    assert is_dominated(a, b, objectives) is True
    assert is_dominated(b, a, objectives) is False


def test_is_dominated_equal_solution_is_not_dominated() -> None:
    objectives = [("score", "maximize"), ("cost", "minimize")]
    a = {"score": 0.8, "cost": 10.0}
    b = {"score": 0.8, "cost": 10.0}

    assert is_dominated(a, b, objectives) is False
    assert is_dominated(b, a, objectives) is False


def test_compute_pareto_front_filters_dominated_rows() -> None:
    df = pd.DataFrame(
        [
            {"clusters_covered": 5, "temporal_std": 10.0, "spatial_mean_km": 100.0},
            {"clusters_covered": 6, "temporal_std": 12.0, "spatial_mean_km": 120.0},
            {"clusters_covered": 6, "temporal_std": 11.0, "spatial_mean_km": 121.0},
            {"clusters_covered": 7, "temporal_std": 13.0, "spatial_mean_km": 130.0},
        ]
    )

    pareto = compute_pareto_front(df)

    assert list(pareto.index) == [3]
