"""Data access layer (canonical).

The goal of this module is to make data loading *boring and predictable*:

- one place to load metadata
- one place to resolve image paths
- one object (`TileSet`) that is passed through the pipeline
"""

from dataselector.data.load import load_tiles
from dataselector.data.tiles import TileSet

__all__ = ["TileSet", "load_tiles"]
