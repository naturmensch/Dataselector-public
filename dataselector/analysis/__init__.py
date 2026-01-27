"""Analysis helpers (metrics, clustering, visualization).

Canonical import paths. Implementations currently re-export `src.*`.
"""

from src.clustering import ClusteringPipeline
from src.metrics import compute_metrics

__all__ = ["ClusteringPipeline", "compute_metrics"]
