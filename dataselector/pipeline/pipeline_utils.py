"""Pipeline helper utilities: derive bounds and candidates between stages.

Functions:
- compute_fine_search_bounds(exploration_pareto_csv): Compute Fine grid bounds from LHS/exploration Pareto
- compute_optuna_bounds(fine_pareto_csv, pct=0.2): Compute Optuna search range from Fine Pareto
- compute_bootstrap_candidates(optuna_csv, delta=5): Compute Bootstrap candidates from Optuna best
- compute_min_distance_km(metadata_csv, strategy='median_nn'): Calculate spatial constraint from metric tile geometry
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _nearest_neighbor_distances_km(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances (km) for projected coordinates."""
    coords = np.column_stack((xs.astype(float), ys.astype(float)))
    if len(coords) < 2:
        return np.array([], dtype=float)

    # Pairwise Euclidean distances in meters, then nearest neighbor in km.
    deltas = coords[:, None, :] - coords[None, :, :]
    dists_m = np.sqrt(np.sum(deltas * deltas, axis=2))
    np.fill_diagonal(dists_m, np.inf)
    return dists_m.min(axis=1) / 1000.0


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


def compute_min_distance_km(
    metadata_csv: str,
    strategy: str = "median_nn",
    percentile: int = 50,
    safety_margin: float = 1.0,
) -> float:
    """
    Calculate initial spatial constraint (min_distance_km) from tile geometry.

    Implements a data-driven approach: computes the nearest-neighbor distance
    distribution across all tiles and derives an appropriate spatial constraint.
    Distances are always computed from metric coordinates (default EPSG:25832).

    Parameters
    ----------
    metadata_csv : str
        Path to tile metadata CSV with coordinate columns
        Requires canonical schema:
        - Bounds: 'ul_x', 'ul_y', 'lr_x', 'lr_y'
        - (optional) precomputed 'center_x', 'center_y'
    strategy : str
        'median_nn' (default): Use median of nearest-neighbor distances
        'percentile_nn': Use given percentile of nearest-neighbor distances
        'mean_nn': Use mean of nearest-neighbor distances
    percentile : int
        Percentile to use when strategy='percentile_nn' (default: 50, i.e., median)
    safety_margin : float
        Multiplier to apply to computed distance (default: 1.0, no margin)

    Returns
    -------
    float
        Computed min_distance_km, rounded to 0.5 km

    Example
    -------
    >>> min_dist = compute_min_distance_km("data/new_all_tiles.csv", strategy="median_nn")
    >>> print(f"Computed min_distance: {min_dist} km")
    Computed min_distance: 45.0 km  # current canonical dataset snapshot (2026-02-09)
    """
    p = Path(metadata_csv)
    if not p.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {p}")

    # Import lazily to avoid adding heavy dependencies for unrelated commands.
    from dataselector.data.io import get_metric_gdf, load_metadata

    df = load_metadata(
        str(p),
        resolve_images=False,
        strict_metric_crs=True,
        metric_epsg=25832,
    )
    gdf_metric = get_metric_gdf(df)
    if gdf_metric is None or not all(
        col in gdf_metric.columns for col in ("_proj_x", "_proj_y")
    ):
        raise RuntimeError(
            "Metric CRS projection missing. compute_min_distance_km requires "
            "projected coordinates (_proj_x/_proj_y)."
        )

    xs = gdf_metric["_proj_x"].to_numpy(dtype=float)
    ys = gdf_metric["_proj_y"].to_numpy(dtype=float)

    if len(xs) < 2:
        print(
            "⚠️  Dataset too small for statistical analysis. "
            "Returning default 28.0 km"
        )
        return 28.0

    nn_distances = _nearest_neighbor_distances_km(xs, ys)
    if len(nn_distances) == 0:
        return 28.0

    # Compute statistic based on strategy
    if strategy == "median_nn":
        computed_dist = float(np.median(nn_distances))
        strategy_name = "median"
    elif strategy == "percentile_nn":
        computed_dist = float(np.percentile(nn_distances, percentile))
        strategy_name = f"{percentile}th percentile"
    elif strategy == "mean_nn":
        computed_dist = float(np.mean(nn_distances))
        strategy_name = "mean"
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be one of: 'median_nn', 'percentile_nn', 'mean_nn'"
        )

    # Apply safety margin
    min_dist = computed_dist * safety_margin

    # Round to nearest 0.5 km
    min_dist_rounded = round(min_dist * 2) / 2

    source_crs = df.attrs.get("source_crs")
    metric_crs = df.attrs.get("metric_crs") or "EPSG:25832"
    transform_applied = bool(df.attrs.get("transform_applied"))
    print(
        "📏 Spatial Constraint Calculation (metric CRS):\n"
        f"   • source_crs={source_crs}, metric_crs={metric_crs}, "
        f"transform_applied={transform_applied}\n"
        f"   • Dataset size: {len(nn_distances)} tiles\n"
        f"   • NN distances: min={nn_distances.min():.1f}, "
        f"max={nn_distances.max():.1f}, {strategy_name}={computed_dist:.1f} km\n"
        f"   • Safety margin: {safety_margin}×\n"
        f"   • Result: min_distance_km = {min_dist_rounded} km"
    )

    return min_dist_rounded


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
