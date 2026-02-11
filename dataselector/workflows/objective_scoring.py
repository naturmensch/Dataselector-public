"""Shared objective scoring helpers for scientific optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from dataselector.data.spatial_schema import (
    normalize_spatial_schema,
    spatial_spread as compute_spatial_spread,
)

_EPS = 1e-12


@dataclass(frozen=True)
class ObjectiveScore:
    """Normalized objective components with feasibility metadata."""

    score: float
    raw_score: float
    diversity_norm: float
    spread_norm: float
    infeasible: bool
    feasibility_ratio: float


def compute_baselines(
    *,
    features: np.ndarray,
    metadata,
    metric: str = "euclidean",
) -> tuple[float, float]:
    """Compute dataset-level baselines for normalized objective scores."""
    if len(features) < 2:
        baseline_diversity = 1.0
    else:
        baseline_diversity = float(np.mean(pdist(features, metric=metric)))
    if baseline_diversity <= 0.0:
        baseline_diversity = 1.0

    spatial_meta = normalize_spatial_schema(metadata, require_bounds=True, copy=True)
    baseline_spread = float(
        compute_spatial_spread(spatial_meta, np.arange(len(spatial_meta), dtype=int))
    )
    if baseline_spread <= 0.0:
        baseline_spread = 1.0

    return baseline_diversity, baseline_spread


def normalized_objective(
    *,
    diversity: float,
    spread: float,
    baseline_diversity: float,
    baseline_spread: float,
    n_selected: int,
    target_n: int,
    weight_diversity: float = 0.5,
    weight_spread: float = 0.5,
    infeasible_penalty: float = 0.1,
) -> ObjectiveScore:
    """Compute normalized score with explicit infeasibility penalty."""
    total_w = float(weight_diversity) + float(weight_spread)
    if total_w <= 0.0:
        raise ValueError("Objective weights must sum to a positive value.")
    wd = float(weight_diversity) / total_w
    ws = float(weight_spread) / total_w

    d_norm = float(diversity) / max(float(baseline_diversity), _EPS)
    s_norm = float(spread) / max(float(baseline_spread), _EPS)
    raw = wd * d_norm + ws * s_norm

    infeasible = int(n_selected) < int(target_n)
    ratio = (
        max(0.0, min(1.0, float(n_selected) / float(target_n)))
        if int(target_n) > 0
        else 0.0
    )
    if infeasible:
        penalized = raw * ratio * float(infeasible_penalty)
    else:
        penalized = raw

    return ObjectiveScore(
        score=float(penalized),
        raw_score=float(raw),
        diversity_norm=float(d_norm),
        spread_norm=float(s_norm),
        infeasible=bool(infeasible),
        feasibility_ratio=float(ratio),
    )
