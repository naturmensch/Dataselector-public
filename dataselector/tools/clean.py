"""Workspace cleanup utilities."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Set

from dataselector.cli_decorators import cli_command

ROOT = Path(__file__).resolve().parents[2]  # dataselector/tools/clean.py -> repo root

PROTECTED_PATHS = {
    "data/images",
    "data/archive",
    "data/raw",
    "models",
    "outputs/final_selection",
    "outputs/kdr100_selection",
}


def sizeof_fmt(num: int, suffix="B") -> str:
    """Format bytes to human readable string."""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def dir_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except Exception:
                pass
    return total


def is_protected(path: Path, extra_protected: Set[str] | None = None) -> bool:
    """Check if path is in protected set.

    Args:
        path: Path to check
        extra_protected: Additional protected paths

    Returns:
        True if path is protected
    """
    protected = PROTECTED_PATHS.copy()
    if extra_protected:
        protected.update(extra_protected)

    rel_path = str(path.relative_to(ROOT))
    for prot in protected:
        if rel_path.startswith(prot):
            return True
    return False


@cli_command(
    "clean-workspace",
    help="Clean workspace files",
    args={
        "delete_outputs": {
            "type": bool,
            "action": "store_true",
            "help": "Delete outputs/ (except protected)",
        },
        "delete_cache": {
            "type": bool,
            "action": "store_true",
            "help": "Delete __pycache__ and .pytest_cache",
        },
        "delete_venvs": {
            "type": bool,
            "action": "store_true",
            "help": "Delete .venv and venv",
        },
        "archive": {
            "type": str,
            "default": None,
            "help": "Archive instead of delete (path to .tar.gz)",
        },
        "yes": {
            "type": bool,
            "action": "store_true",
            "help": "Actually perform cleanup (default is dry-run)",
        },
    },
)
def clean_workspace(
    delete_outputs: bool = False,
    delete_cache: bool = False,
    delete_venvs: bool = False,
    archive: str | None = None,
    yes: bool = False,
    extra_protected: Set[str] | None = None
) -> int:
    """Clean workspace by removing or archiving old files.

    Args:
        delete_outputs: Delete outputs/ directory (except protected)
        delete_cache: Delete __pycache__ and .pytest_cache
        delete_venvs: Delete .venv and venv directories
        archive: If provided, archive instead of delete
        yes: If False (default), only show what would be done (dry-run)
        extra_protected: Additional paths to protect

    Returns:
        0 on success
    """
    archive_path = archive
    dry_run = not yes  # Invert: --yes means NOT dry-run
    to_remove = []

    # Collect candidates
    if delete_outputs:
        outputs_dir = ROOT / "outputs"
        if outputs_dir.exists():
            for item in outputs_dir.iterdir():
                if not is_protected(item, extra_protected):
                    to_remove.append(item)

    if delete_cache:
        for cache_dir in ROOT.rglob("__pycache__"):
            to_remove.append(cache_dir)
        for cache_dir in ROOT.rglob(".pytest_cache"):
            to_remove.append(cache_dir)
        for cache_dir in ROOT.rglob(".mypy_cache"):
            to_remove.append(cache_dir)

    if delete_venvs:
        for venv in [ROOT / ".venv", ROOT / "venv"]:
            if venv.exists():
                to_remove.append(venv)

    if not to_remove:
        print("No files to clean.")
        return 0

    # Calculate total size
    total_size = sum(dir_size(p) if p.is_dir() else p.stat().st_size for p in to_remove)

    print(f"Found {len(to_remove)} items to clean ({sizeof_fmt(total_size)}):")
    for item in to_remove[:10]:
        rel_path = item.relative_to(ROOT)
        item_size = dir_size(item) if item.is_dir() else item.stat().st_size
        print(f"  - {rel_path} ({sizeof_fmt(item_size)})")
    if len(to_remove) > 10:
        print(f"  ... and {len(to_remove) - 10} more")

    if dry_run:
        print("\n[DRY RUN] No files were removed. Use --yes to proceed.")
        return 0

    # Archive or delete
    if archive_path:
        archive_path_obj = Path(archive_path)
        archive_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path_obj, "w:gz") as tar:
            for item in to_remove:
                tar.add(item, arcname=item.relative_to(ROOT))
        print(f"\nArchived {len(to_remove)} items to: {archive_path_obj}")
    else:
        for item in to_remove:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"\nDeleted {len(to_remove)} items ({sizeof_fmt(total_size)})")

    return 0
