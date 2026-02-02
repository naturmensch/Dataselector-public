"""Pipeline helper utilities: derive bounds and candidates between stages.

Functions:
- compute_fine_search_bounds(exploration_pareto_csv): Compute Fine grid bounds from LHS/exploration Pareto
- compute_optuna_bounds(fine_pareto_csv, pct=0.2): Compute Optuna search range from Fine Pareto
- compute_bootstrap_candidates(optuna_csv, delta=5): Compute Bootstrap candidates from Optuna best
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def compute_fine_search_bounds(exploration_pareto_csv: str) -> List[float]:
    """Compute Fine sweep distance bounds from exploration (LHS or legacy Coarse) Pareto.

    Args:
        exploration_pareto_csv: Path to pareto_solutions.csv from LHS or Coarse exploration

    Returns:
        Five-point grid around median distance: [center-10, center-5, center, center+5, center+10]
    """
    p = Path(exploration_pareto_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # Choose center as median of min_distance_km among best Pareto candidates
    if "min_distance_km" in df.columns:
        center = float(df["min_distance_km"].median())
    else:
        center = 40.0
    # produce five-point fine grid centered on center
    vals = [
        max(10.0, center - 10),
        center - 5,
        center,
        center + 5,
        min(100.0, center + 10),
    ]
    return [float(round(v, 1)) for v in vals]


def compute_optuna_bounds(fine_pareto_csv: str, pct: float = 0.2) -> Tuple[int, int]:
    p = Path(fine_pareto_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "min_distance_km" in df.columns:
        center = float(df["min_distance_km"].median())
    else:
        center = 40.0
    lo = max(10, int(center * (1 - pct)))
    hi = min(100, int(center * (1 + pct)))
    return lo, hi


def compute_bootstrap_candidates(optuna_csv: str, delta: int = 5) -> List[int]:
    p = Path(optuna_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "user_attrs_min_distance_km" in df.columns:
        center = int(
            df.sort_values("user_attrs_min_distance_km", ascending=False).iloc[0][
                "user_attrs_min_distance_km"
            ]
        )
    elif "params_min_distance_km" in df.columns:
        center = int(
            df.sort_values("params_min_distance_km", ascending=False).iloc[0][
                "params_min_distance_km"
            ]
        )
    else:
        center = 40
    return [center - delta, center, center + delta]


def compute_adaptive_n_initial(
    n_dimensions: int, n_tiles: int = None, strategy: str = "modern"
) -> int:
    """Compute an adaptive initial sample size for the exploration phase.

    Strategies:
      - 'modern': use a dimension-aware rule of thumb (default: max(2*D^2, 20))
      - 'legacy': keep compatibility with LHS/Taguchi legacy: max(27, sqrt(n_tiles)) if n_tiles known

    Args:
        n_dimensions: number of optimization dimensions (e.g., 3 weights)
        n_tiles: optional dataset size used by legacy rule
        strategy: 'modern' or 'legacy'

    Returns:
        integer number of initial samples
    """
    if strategy == "modern":
        # A conservative SOTA rule-of-thumb; ensures at least 20 samples for small D
        return max(2 * (int(n_dimensions) ** 2), 20)
    elif strategy == "legacy":
        if n_tiles is None:
            return 27
        try:
            return max(27, int(int(n_tiles) ** 0.5))
        except Exception:
            return 27
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
