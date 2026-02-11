from __future__ import annotations

from dataclasses import dataclass

from dataselector.workflows.optuna_autoscale import _select_best_production_trial


@dataclass
class _Trial:
    value: float
    user_attrs: dict
    params: dict | None = None


@dataclass
class _Study:
    trials: list
    best_trial: _Trial


def test_select_best_production_trial_prefers_non_diagnostic() -> None:
    diagnostic = _Trial(value=100.0, user_attrs={"full_coverage_mode": True})
    production = _Trial(value=50.0, user_attrs={"full_coverage_mode": False})
    study = _Study(trials=[diagnostic, production], best_trial=diagnostic)

    best, from_production = _select_best_production_trial(study)
    assert from_production is True
    assert best is production


def test_select_best_production_trial_falls_back_when_only_diagnostic() -> None:
    diagnostic = _Trial(value=100.0, user_attrs={"full_coverage_mode": True})
    study = _Study(trials=[diagnostic], best_trial=diagnostic)

    best, from_production = _select_best_production_trial(study)
    assert from_production is False
    assert best is diagnostic
