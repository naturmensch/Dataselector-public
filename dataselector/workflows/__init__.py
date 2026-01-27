"""Workflow entrypoints.

Best practice separation:
- `dataselector.*` contains reusable library code.
- `dataselector.workflows.*` contains orchestration (runs, artifacts).

Initially, these workflows are thin wrappers around the existing scripts.
That keeps 100% functionality while we reorganize.
"""

__all__ = []
