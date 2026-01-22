import pytest
import numpy as np
from pathlib import Path

from tests._helpers.load_script import load_script


@pytest.fixture(autouse=True)
def skip_if_no_torch():
    pytest.importorskip("torch")


@pytest.fixture(scope="module")
def uq_mod():
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(ROOT / "scripts" / "uncertainty_quantification.py", module_name="scripts.uncertainty_quantification_test")


@pytest.fixture
def train_ensemble(uq_mod):
    return uq_mod.train_ensemble


@pytest.fixture
def predict_with_uncertainty(uq_mod):
    return uq_mod.predict_with_uncertainty


def test_train_predict_ensemble_small(train_ensemble, predict_with_uncertainty):
    X = np.random.RandomState(0).randn(30, 4)
    y = X[:, 0] * 0.5 + X[:, 1] * -0.2 + np.random.randn(30) * 0.1
    models = train_ensemble(X, y, n_models=3, epochs=5, lr=1e-2)
    mean, std = predict_with_uncertainty(models, X[:5])
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std >= 0)
