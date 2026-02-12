from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataselector.workflows.optuna_autoscale import _select_plateau_feasible_trial


def _trial(
    *,
    value: float,
    n_samples: int,
    n_selected: int,
    infeasible: bool = False,
    diagnostic: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        value=value,
        params={"a": 0.4, "b": 0.3, "c": 0.3, "min_distance_km": 28},
        user_attrs={
            "n_samples": n_samples,
            "n_selected": n_selected,
            "infeasible": infeasible,
            "full_coverage_mode": diagnostic,
        },
    )


def test_plateau_selection_picks_smallest_feasible_within_band() -> None:
    study = SimpleNamespace(
        trials=[
            _trial(value=1.000, n_samples=40, n_selected=40),
            _trial(value=0.990, n_samples=34, n_selected=34),
            _trial(value=0.965, n_samples=27, n_selected=27),
            _trial(value=1.050, n_samples=24, n_selected=22, infeasible=True),
        ]
    )
    selected, from_prod, meta = _select_plateau_feasible_trial(
        study=study,
        plateau_delta=0.02,
        strict_feasible_selection=True,
    )

    assert from_prod is True
    assert int(selected.user_attrs["n_samples"]) == 34
    assert meta["rule"] == "minimal_feasible_plateau"
    assert meta["selected_n_samples"] == 34


def test_plateau_selection_fails_when_no_feasible_and_strict() -> None:
    study = SimpleNamespace(
        trials=[
            _trial(value=1.0, n_samples=34, n_selected=20, infeasible=True),
            _trial(value=0.9, n_samples=40, n_selected=10, infeasible=True),
        ]
    )

    with pytest.raises(RuntimeError, match="No feasible production trials"):
        _select_plateau_feasible_trial(
            study=study,
            plateau_delta=0.02,
            strict_feasible_selection=True,
        )
