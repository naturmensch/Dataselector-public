        models.append(model)

    return models


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
