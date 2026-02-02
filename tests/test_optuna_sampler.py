from pathlib import Path

import pytest

from tests._helpers.load_script import load_script

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_optuna():
    pytest.importorskip("optuna")


@pytest.fixture(scope="module")
def optuna_optimize_mod():
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(
        ROOT / "scripts" / "optuna_optimize.py", module_name="scripts.optuna_optimize"
    )


@pytest.fixture
def get_optuna_sampler(optuna_optimize_mod):
    return optuna_optimize_mod.get_optuna_sampler


def test_get_optuna_sampler_qmc(get_optuna_sampler):
    sampler = get_optuna_sampler("qmc", seed=123)
    # Check that sampler exposes the expected class name (QMCSampler) or falls back to TPE
    assert sampler is not None
    name = sampler.__class__.__name__.lower()
    assert "qmcsampler" in name or "tpesampler" in name


def test_get_optuna_sampler_cmaes(get_optuna_sampler):
    sampler = get_optuna_sampler("cmaes", seed=123)
    assert sampler is not None
    assert sampler.__class__.__name__.lower() == "cmaessampler"


def test_get_optuna_sampler_tpe(get_optuna_sampler):
    sampler = get_optuna_sampler("tpe", seed=123)
    assert sampler is not None
    assert sampler.__class__.__name__.lower() == "tpesampler"


# More targeted regression tests for sampler API compatibility
class FakeSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeSamplerRejectQMC:
    def __init__(self, **kwargs):
        if "qmc_type" in kwargs or "qmc" in kwargs:
            raise TypeError("unsupported kwarg")
        self.kwargs = kwargs


class AlwaysErrorSampler:
    def __init__(self, **kwargs):
        raise TypeError("nope")


class DummyTPE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_qmc_accepts_qmc_type(monkeypatch, get_optuna_sampler):
    import optuna

    monkeypatch.setattr(optuna.samplers, "QMCSampler", FakeSampler)

    s = get_optuna_sampler("qmc", seed=1)
    assert isinstance(s, FakeSampler)
    # Should have been called with qmc_type='sobol'
    assert "qmc_type" in s.kwargs or "qmc" in s.kwargs


def test_qmc_fallback_to_qmc_kw(monkeypatch, get_optuna_sampler):
    import optuna

    monkeypatch.setattr(optuna.samplers, "QMCSampler", FakeSamplerRejectQMC)

    s = get_optuna_sampler("qmc", seed=1)
    assert isinstance(s, FakeSamplerRejectQMC)


def test_qmc_all_rejects_fall_back_to_tpe(monkeypatch, get_optuna_sampler):
    import optuna

    monkeypatch.setattr(optuna.samplers, "QMCSampler", AlwaysErrorSampler)
    monkeypatch.setattr(optuna.samplers, "TPESampler", DummyTPE)

    s = get_optuna_sampler("qmc", seed=1)
    assert isinstance(s, DummyTPE)
