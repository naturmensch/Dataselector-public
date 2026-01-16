import pytest
pytest.importorskip('torch')
import numpy as np
from scripts.uncertainty_quantification import train_ensemble, predict_with_uncertainty


def test_train_predict_ensemble_small():
    X = np.random.RandomState(0).randn(30, 4)
    y = (X[:, 0] * 0.5 + X[:, 1] * -0.2 + np.random.randn(30) * 0.1)
    models = train_ensemble(X, y, n_models=3, epochs=5, lr=1e-2)
    mean, std = predict_with_uncertainty(models, X[:5])
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std >= 0)
