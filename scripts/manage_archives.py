#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Optional
import fnmatch
import time


def _matches_any(patterns: List[str], rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def archive_outputs(src: Path, dst: Path, exclude: Optional[List[str]] = None) -> Path:
    """
    Create a gzipped tar archive of the given outputs directory under dst.

    - Preserves an "outputs/" root inside the tarball.
    - Excludes any files matching patterns provided via `exclude`.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"outputs dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_path = dst / f"outputs_{ts}.tar.gz"
    patterns = [p.strip() for p in (exclude or []) if p and p.strip()]

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(src).as_posix()
            if patterns and _matches_any(patterns, rel):
                continue
            arcname = Path("outputs") / rel
            tar.add(f, arcname=arcname.as_posix())
    return archive_path


def list_archives(dst: Path) -> List[Path]:
    dst = Path(dst)
    if not dst.exists():
        return []
    return sorted(p for p in dst.glob("outputs_*.tar.gz") if p.is_file())


def restore_archive(archive_path: Path, target_dir: Path) -> None:
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)


__all__ = ["archive_outputs", "list_archives", "restore_archive"]
