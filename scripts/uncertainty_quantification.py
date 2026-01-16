"""Lightweight Deep Ensemble utilities for uncertainty-aware validation.

This module provides convenience wrappers to train shallow MLP ensembles on
bootstrap-style training data (X: hyperparameters, y: metrics) and predict mean
and epistemic std as UQ estimates.

Note: This is a pragmatic, minimal implementation intended as a faster
alternative to running hundreds of resamples at prediction time. It still
requires generating a modest training set (e.g., 30-100 resamples) but the
inference is very fast once the ensemble is trained.
"""
from typing import List, Tuple, Sequence
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


def train_ensemble(X: np.ndarray, y: np.ndarray, n_models: int = 5, epochs: int = 50, lr: float = 1e-3) -> List[EnsembleMLP]:
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
        models.append(model)

    return models


def predict_with_uncertainty(models: Sequence[EnsembleMLP], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def fit_ensemble_on_bootstrap_df(df_boot: np.ndarray, input_cols: Sequence[str], target_col: str, n_models: int = 5, epochs: int = 50):
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
