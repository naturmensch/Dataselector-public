"""Dataselector (public API).

This package is the **new stable import path** for the project.

Today it mainly re-exports the existing implementation that still lives in
`src/` (compatibility). Over time, code will be moved from `src/` into this
package without breaking users.

Note: Heavy dependencies (torch, sklearn, etc.) are lazy-loaded to support
lightweight CLI tools without requiring full environment.
"""

from __future__ import annotations

# NOTE: Keep this version in sync with the legacy src package for now.
__version__ = "0.1.0"

# Lazy-loaded public API - modules are imported when accessed
__all__ = [
    "__version__",
    # legacy re-exports (lazy)
    "MetadataProcessor",
    "FeatureExtractor",
    "ClusteringPipeline",
    "DiversitySelector",
    "Visualizer",
    "MultiCriteriaFacilityLocation",
    "SpatialConstrainedFacilityLocation",
    "compute_metrics",
    # new API (lazy)
    "load_tiles",
    "load_or_compute_features",
]


def __getattr__(name: str):
    """Lazy import of heavy dependencies.
    
    This allows lightweight tools (check, archive, etc.) to import
    from dataselector.tools without requiring torch, sklearn, etc.
    """
    if name == "FeatureExtractor":
        from dataselector.features.feature_extractor import FeatureExtractor
        return FeatureExtractor
    
    if name == "MetadataProcessor":
        from dataselector.data.metadata_processor import MetadataProcessor
        return MetadataProcessor
    
    if name == "Visualizer":
        from dataselector.analysis.visualizer import Visualizer
        return Visualizer
    
    if name == "ClusteringPipeline":
        from dataselector.analysis import ClusteringPipeline
        return ClusteringPipeline
    
    if name == "compute_metrics":
        from dataselector.analysis import compute_metrics
        return compute_metrics
    
    if name == "DiversitySelector":
        from dataselector.selection import DiversitySelector
        return DiversitySelector
    
    if name == "MultiCriteriaFacilityLocation":
        from dataselector.selection import MultiCriteriaFacilityLocation
        return MultiCriteriaFacilityLocation
    
    if name == "SpatialConstrainedFacilityLocation":
        from dataselector.selection import SpatialConstrainedFacilityLocation
        return SpatialConstrainedFacilityLocation
    
    if name == "load_tiles":
        from dataselector.data.load import load_tiles
        return load_tiles
    
    if name == "load_or_compute_features":
        from dataselector.features.pipeline import load_or_compute_features
        return load_or_compute_features
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
