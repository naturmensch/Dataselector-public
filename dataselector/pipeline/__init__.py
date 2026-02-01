"""High-level pipelines.

This layer is intentionally small:
- It provides a simple pipeline entrypoint (useful for onboarding / quick runs).
- The thesis-grade orchestration lives under workflows.
"""

from src.main import KDR100SelectionPipeline

__all__ = ["KDR100SelectionPipeline"]
