"""High-level pipeline package.

Keep imports lazy so submodules like :mod:`dataselector.pipeline.cache` can be
imported without forcing heavy runtime dependencies from ``main.py``.
"""

from __future__ import annotations

__all__ = ["KDR100SelectionPipeline"]


def __getattr__(name: str):
    if name == "KDR100SelectionPipeline":
        from dataselector.pipeline.main import KDR100SelectionPipeline

        return KDR100SelectionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
