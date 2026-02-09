from __future__ import annotations

from pathlib import Path

CANONICAL_METADATA_RELATIVE_PATH = Path("data/new_all_tiles.csv")


def _resolve_root(root: str | Path | None = None) -> Path:
    return Path.cwd() if root is None else Path(root)


def canonical_metadata_path(root: str | Path | None = None) -> Path:
    """Return canonical metadata path for production workflows."""
    return (_resolve_root(root) / CANONICAL_METADATA_RELATIVE_PATH).resolve()


def _resolve_input_path(path: str | Path, root: str | Path | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _resolve_root(root) / p
    return p.resolve()


def assert_canonical_metadata(
    path: str | Path | None,
    *,
    context: str,
    root: str | Path | None = None,
) -> Path:
    """Validate metadata source path against canonical production source.

    Returns canonical path (absolute, resolved). Raises ValueError on mismatch.
    """
    canonical = canonical_metadata_path(root)

    if path is None:
        return canonical

    resolved = _resolve_input_path(path, root)
    if resolved != canonical:
        raise ValueError(
            f"{context}: invalid metadata source '{path}'. "
            f"Only '{CANONICAL_METADATA_RELATIVE_PATH.as_posix()}' is allowed "
            "for production workflows."
        )

    return canonical
