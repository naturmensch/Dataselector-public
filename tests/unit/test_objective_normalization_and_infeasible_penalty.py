from __future__ import annotations

import numpy as np
import pandas as pd

from dataselector.workflows.objective_scoring import (
    compute_baselines,
    normalized_objective,
)


def test_compute_baselines_returns_positive_values() -> None:
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=float)
    metadata = pd.DataFrame(
        {
            "ul_x": [0.0, 10.0, 20.0],
            "ul_y": [30.0, 30.0, 30.0],
            "lr_x": [1.0, 11.0, 21.0],
            "lr_y": [29.0, 29.0, 29.0],
        }
    )
    d_base, s_base = compute_baselines(features=features, metadata=metadata)
    assert d_base > 0.0
    assert s_base > 0.0


def test_normalized_objective_penalizes_infeasible_trials() -> None:
    feasible = normalized_objective(
        diversity=2.0,
        spread=3.0,
        baseline_diversity=2.0,
        baseline_spread=3.0,
        n_selected=24,
        target_n=24,
        infeasible_penalty=0.1,
    )
    infeasible = normalized_objective(
        diversity=2.0,
        spread=3.0,
        baseline_diversity=2.0,
        baseline_spread=3.0,
        n_selected=12,
        target_n=24,
        infeasible_penalty=0.1,
    )
    assert feasible.infeasible is False
    assert infeasible.infeasible is True
    assert feasible.score > infeasible.score
    assert infeasible.feasibility_ratio == 0.5
