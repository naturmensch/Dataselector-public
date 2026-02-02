"""Sampling strategies utilities (QMC / LHS wrappers).

Provides small convenience functions used by the exploration phase
(e.g., sampling weight combinations on the simplex for phase 1).
"""

from typing import List, Tuple

import numpy as np

try:
    from scipy.stats import qmc

    HAS_QMC = True
except Exception:
    HAS_QMC = False


def sample_unit_hypercube_sobol(
    n_samples: int, n_params: int, seed: int = 42
) -> np.ndarray:
    """Generate Sobol (QMC) samples in the unit hypercube [0,1]^d.

    Args:
        n_samples: number of samples to draw
        n_params: dimensionality d
        seed: optional scramble seed

    Returns:
        numpy array of shape (n_samples, n_params) with values in [0,1]
    """
    if not HAS_QMC:
        raise RuntimeError("scipy.stats.qmc (Sobol) not available")

    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    return sampler.random(n_samples)


def sample_unit_hypercube_lhs(
    n_samples: int, n_params: int, seed: int = 42
) -> np.ndarray:
    """Generate Latin Hypercube samples in the unit hypercube [0,1]^d.

    Args:
        n_samples: number of samples to draw
        n_params: dimensionality d
        seed: random seed

    Returns:
        numpy array of shape (n_samples, n_params) with values in [0,1]
    """
    if not HAS_QMC:
        raise RuntimeError("scipy.stats.qmc (LHS) not available")

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    return sampler.random(n=n_samples)


def sample_weights_on_simplex_sobol(
    n_points: int = 50, dim: int = 3, seed: int = 42
) -> List[Tuple[float, ...]]:
    """Generate weight combinations on the (dim-1)-simplex using Sobol sampling.

    The unit-cube samples are projected to the simplex by normalizing each row
    to sum to 1. This mirrors the existing approach for LHS-based weights.
    """
    samples = sample_unit_hypercube_sobol(n_points, dim, seed=seed)
    weights = samples / samples.sum(axis=1)[:, None]
    return [tuple(w) for w in weights]


def sample_weights_on_simplex_lhs(
    n_points: int = 50, dim: int = 3, seed: int = 42
) -> List[Tuple[float, ...]]:
    """Generate weight combinations on the (dim-1)-simplex using LHS sampling."""
    samples = sample_unit_hypercube_lhs(n_points, dim, seed=seed)
    weights = samples / samples.sum(axis=1)[:, None]
    return [tuple(w) for w in weights]
