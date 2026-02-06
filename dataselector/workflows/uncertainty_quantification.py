"""Uncertainty Quantification helpers using deep ensembles."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    """Simple neural network regressor used for ensemble members."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_models: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    random_seed: int = 42,
) -> List[nn.Module]:
    """Train an ensemble of regressors on numpy arrays."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    x_norm = (X - x_mean) / x_std

    x_tensor = torch.from_numpy(x_norm)
    y_tensor = torch.from_numpy(y)

    models: List[nn.Module] = []
    for i in range(n_models):
        torch.manual_seed(random_seed + i)

        model = SimpleRegressor(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        # Persist normalization parameters on model object.
        model.X_mean = x_mean
        model.X_std = x_std
        models.append(model)

    return models


def fit_ensemble_on_bootstrap_df(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    n_models: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    random_seed: int = 42,
) -> List[nn.Module]:
    """Train an ensemble from bootstrap dataframe columns."""
    X = df[input_cols].values
    y = df[target_col].values
    return train_ensemble(
        X,
        y,
        n_models=n_models,
        epochs=epochs,
        lr=lr,
        random_seed=random_seed,
    )


def predict_with_uncertainty(
    models: List[nn.Module], X_query: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean/std uncertainty for query samples."""
    if not models:
        return np.array([]), np.array([])

    X_query = np.asarray(X_query, dtype=np.float32)
    per_model = []
    for model in models:
        model.eval()
        with torch.no_grad():
            x_norm = (X_query - model.X_mean) / model.X_std
            x_tensor = torch.from_numpy(x_norm)
            pred = model(x_tensor).cpu().numpy().reshape(-1)
            per_model.append(pred)

    stacked = np.vstack(per_model)
    return stacked.mean(axis=0), stacked.std(axis=0)
