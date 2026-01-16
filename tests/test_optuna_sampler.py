import pytest
pytest.importorskip('optuna')
from scripts.optuna_optimize import get_optuna_sampler


def test_get_optuna_sampler_qmc():
    sampler = get_optuna_sampler('qmc', seed=123)
    # Check that sampler exposes the expected class name (QMCSampler) or falls back to TPE
    assert sampler is not None
    name = sampler.__class__.__name__.lower()
    assert 'qmcsampler' in name or 'tpesampler' in name


def test_get_optuna_sampler_cmaes():
    sampler = get_optuna_sampler('cmaes', seed=123)
    assert sampler is not None
    assert sampler.__class__.__name__.lower() == 'cmaessampler'


def test_get_optuna_sampler_tpe():
    sampler = get_optuna_sampler('tpe', seed=123)
    assert sampler is not None
    assert sampler.__class__.__name__.lower() == 'tpesampler'
