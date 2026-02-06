"""Administrative & maintenance tools for Dataselector.

This module provides CLI tools for:
- Checking protected files and environment usage
- Verifying and managing archives
- Auditing spatial alignment
- Workspace cleanup
- Documentation link maintenance

Usage:
    python -m dataselector tools check-protected
    python -m dataselector tools verify-archive
    python -m dataselector tools align-audit

Note: Tools are lazy-loaded on demand to avoid heavy dependencies for CLI usage.
"""

# Lazy imports - modules are imported when accessed
__all__ = ["check", "archive", "audit", "clean", "docs_link"]


def __getattr__(name: str):
    """Lazy import of tools modules."""
    if name in __all__:
        import importlib

        return importlib.import_module(f"dataselector.tools.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
