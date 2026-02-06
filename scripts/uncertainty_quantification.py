"""Uncertainty Quantification using Deep Ensembles.

This module provides functions for fitting ensemble models on bootstrap data
and predicting with uncertainty estimates.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

        models.append(model)

    return models


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