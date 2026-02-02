"""Dataselector (public API).

This package is the **new stable import path** for the project.

Today it mainly re-exports the existing implementation that still lives in
`src/` (compatibility). Over time, code will be moved from `src/` into this
package without breaking users.
"""

from __future__ import annotations

# NOTE: Keep this version in sync with the legacy src package for now.
__version__ = "0.1.0"

# Public high-level objects (currently implemented in src/)
from dataselector.features.feature_extractor import FeatureExtractor
from dataselector.data.metadata_processor import MetadataProcessor
from dataselector.analysis.visualizer import Visualizer

# Stable import paths for major subdomains
from dataselector.analysis import ClusteringPipeline, compute_metrics
from dataselector.selection import (
    DiversitySelector,
    MultiCriteriaFacilityLocation,
    SpatialConstrainedFacilityLocation,
)

# New “canonical” data/feature entrypoints (thin wrappers around src for now)
from dataselector.data.load import load_tiles
from dataselector.features.pipeline import load_or_compute_features

__all__ = [
    "__version__",
    # legacy re-exports
    "MetadataProcessor",
    "FeatureExtractor",
    "ClusteringPipeline",
    "DiversitySelector",
    "Visualizer",
    "MultiCriteriaFacilityLocation",
    "SpatialConstrainedFacilityLocation",
    "compute_metrics",
    # new API
    "load_tiles",
    "load_or_compute_features",
]

