"""Data access layer (canonical).

The goal of this module is to make data loading *boring and predictable*:

- one place to load metadata
- one place to resolve image paths
- one object (`TileSet`) that is passed through the pipeline
"""

from dataselector.data.load import load_tiles
from dataselector.data.tiles import TileSet

__all__ = ["TileSet", "load_tiles", "build_tiles"]

def __getattr__(name):
    """Lazy import for data utilities."""
    if name == "build_tiles":
        from dataselector.data.build_tiles import build_tiles as _build_tiles
        globals()[name] = _build_tiles
        return _build_tiles
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
