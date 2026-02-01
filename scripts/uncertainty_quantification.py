<<<<<<< HEAD
"""Uncertainty Quantification using Deep Ensembles.

This module provides functions for fitting ensemble models on bootstrap data
and predicting with uncertainty estimates.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


class SimpleRegressor(nn.Module):
    """Simple neural network regressor for ensemble."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def fit_ensemble_on_bootstrap_df(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    n_models: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    random_seed: int = 42
) -> List[nn.Module]:
    """Fit an ensemble of neural networks on bootstrap data.

    Args:
        df: Bootstrap dataframe
        input_cols: Column names for inputs
        target_col: Column name for target
        n_models: Number of models in ensemble
        epochs: Training epochs per model
        lr: Learning rate
        random_seed: Random seed

    Returns:
        List of trained models
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Prepare data
    X = df[input_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    models = []
    for i in range(n_models):
        # Different seed for each model
        torch.manual_seed(random_seed + i)

        model = SimpleRegressor(len(input_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(y)

        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        # Store normalization params in model
        model.X_mean = X_mean
        model.X_std = X_std

=======
"""Lightweight Deep Ensemble utilities for uncertainty-aware validation.

This module provides convenience wrappers to train shallow MLP ensembles on
bootstrap-style training data (X: hyperparameters, y: metrics) and predict mean
and epistemic std as UQ estimates.

Note: This is a pragmatic, minimal implementation intended as a faster
alternative to running hundreds of resamples at prediction time. It still
requires generating a modest training set (e.g., 30-100 resamples) but the
inference is very fast once the ensemble is trained.
"""

from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class EnsembleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_ensemble(
    X: np.ndarray, y: np.ndarray, n_models: int = 5, epochs: int = 50, lr: float = 1e-3
) -> List[EnsembleMLP]:
    """Train an ensemble of small MLPs to predict `y` from `X`.

    Args:
        X: shape (N, D)
        y: shape (N,) or (N,1)
        n_models: number of ensemble members
        epochs: epochs per model
        lr: learning rate

    Returns:
        list of trained models
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available; cannot train ensemble")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    models: List[EnsembleMLP] = []

    for i in range(n_models):
        torch.manual_seed(42 + i)
        model = EnsembleMLP(input_dim=X.shape[1])
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            pred = model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
>>>>>>> ci/add-smoke-tests
        models.append(model)

    return models


<<<<<<< HEAD
<<<<<<< HEAD
def predict_with_uncertainty(
    models: List[nn.Module],
    X_query: np.ndarray
) -> Tuple[float, float]:
    """Predict with uncertainty using ensemble.

    Args:
        models: List of trained models
        X_query: Query input (shape: [1, n_features])

    Returns:
        Tuple of (mean_prediction, std_prediction)
    """
    if not models:
        return float('nan'), float('nan')

    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            # Normalize input
            X_norm = (X_query - model.X_mean) / model.X_std
            X_tensor = torch.from_numpy(X_norm.astype(np.float32))
            pred = model(X_tensor).item()
            predictions.append(pred)

    if not predictions:
        return float('nan'), float('nan')

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)

    return mean_pred, std_pred
=======
def predict_with_uncertainty(models: Sequence[EnsembleMLP], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
=======
def predict_with_uncertainty(
    models: Sequence[EnsembleMLP], X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
>>>>>>> chore/ci-lint-attrs-gdf
    """Return mean and std predictions across ensemble members.

    Args:
        models: trained ensemble
        X: array shape (M, D)

    Returns:
        mean: (M,), std: (M,)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available; cannot predict with ensemble")

    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(X_t).numpy().ravel()
            preds.append(p)
    arr = np.stack(preds, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def fit_ensemble_on_bootstrap_df(
    df_boot: np.ndarray,
    input_cols: Sequence[str],
    target_col: str,
    n_models: int = 5,
    epochs: int = 50,
):
    """Fit ensemble to bootstrap DataFrame exported from `bootstrap_candidate`.

    Args:
        df_boot: pandas DataFrame-like (numpy structured) or dict-like; we only use numpy here
    """
    # Lightweight: accept numpy recarray / structured dtype or a simple 2D numpy
    import pandas as pd

    df = pd.DataFrame(df_boot)
    X = df[list(input_cols)].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    models = train_ensemble(X, y, n_models=n_models, epochs=epochs)
    return models
>>>>>>> ci/add-smoke-tests
