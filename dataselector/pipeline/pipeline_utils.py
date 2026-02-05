"""Pipeline helper utilities: derive bounds and candidates between stages.

Functions:
- compute_fine_search_bounds(exploration_pareto_csv): Compute Fine grid bounds from LHS/exploration Pareto
- compute_optuna_bounds(fine_pareto_csv, pct=0.2): Compute Optuna search range from Fine Pareto
- compute_bootstrap_candidates(optuna_csv, delta=5): Compute Bootstrap candidates from Optuna best
- compute_min_distance_km(metadata_csv, strategy='median_nn'): Calculate spatial constraint from tile geometry
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
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

    Parameters
    ----------
    metadata_csv : str
        Path to tile metadata CSV with coordinate columns
        Accepts either:
        - WGS84: 'N' (latitude) and 'left' (longitude)
        - UTM EPSG:3857: 'ul_x', 'ul_y' (upper-left) or 'lr_x', 'lr_y' (lower-right)
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
    >>> print(f"Optimal min_distance: {min_dist} km")
    Optimal min_distance: 28.5 km
    """
    p = Path(metadata_csv)
    if not p.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {p}")

    df = pd.read_csv(p)

    # Extract coordinates (try different coordinate systems)
    if "N" in df.columns and "left" in df.columns:
        # WGS84 lat/lon
        lats = df["N"].values
        lons = df["left"].values
        is_utm = False
    elif "ul_x" in df.columns and "ul_y" in df.columns:
        # UTM EPSG:3857 (upper-left corner as tile center proxy)
        xs = df["ul_x"].values
        ys = df["ul_y"].values
        is_utm = True
    elif "lr_x" in df.columns and "lr_y" in df.columns:
        # UTM EPSG:3857 (lower-right corner as tile center proxy)
        xs = df["lr_x"].values
        ys = df["lr_y"].values
        is_utm = True
    else:
        raise ValueError(
            f"CSV must contain either 'N'/'left' (WGS84) or 'ul_x'/'ul_y' (UTM) columns. "
            f"Found: {list(df.columns)}"
        )

    if (is_utm and len(xs) < 2) or (not is_utm and len(lats) < 2):
        print(
            f"⚠️  Dataset too small for statistical analysis. "
            "Returning default 28.0 km"
        )
        return 28.0

    # Compute nearest-neighbor distances
    nn_distances = []

    if is_utm:
        # UTM coordinates: Euclidean distance in meters
        for i in range(len(xs)):
            x1, y1 = xs[i], ys[i]
            distances = []
            for j in range(len(xs)):
                if i != j:
                    x2, y2 = xs[j], ys[j]
                    dist_meters = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    dist_km = dist_meters / 1000.0
                    distances.append(dist_km)
            if distances:
                nn_distance = min(distances)
                nn_distances.append(nn_distance)
    else:
        # WGS84: Haversine distance
        for i in range(len(lats)):
            lat1, lon1 = lats[i], lons[i]
            distances = []
            for j in range(len(lats)):
                if i != j:
                    lat2, lon2 = lats[j], lons[j]
                    # Approximate km per degree (ignoring spheroid for speed)
                    dlat_km = (lat2 - lat1) * 111.0
                    dlon_km = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
                    dist_km = np.sqrt(dlat_km**2 + dlon_km**2)
                    distances.append(dist_km)
            if distances:
                nn_distance = min(distances)
                nn_distances.append(nn_distance)

    nn_distances = np.array(nn_distances)

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

    coord_system = "UTM EPSG:3857" if is_utm else "WGS84 (lat/lon)"
    print(
        f"📏 Spatial Constraint Calculation ({coord_system}):\n"
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
