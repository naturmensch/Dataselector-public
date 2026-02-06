"""Analysis helpers (metrics, clustering, visualization).

Canonical import paths. Implementations currently re-export `src.*`.
"""

from dataselector.analysis.metrics import compute_metrics
from dataselector.selection.clustering import ClusteringPipeline

__all__ = ["ClusteringPipeline", "compute_metrics"]
