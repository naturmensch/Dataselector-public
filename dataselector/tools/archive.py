"""Archive verification and management tools."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tarfile
import re
import sys

from dataselector.cli_decorators import cli_command

ROOT = Path(__file__).resolve().parents[2]  # dataselector/tools/archive.py -> repo root

# ===== Archive Verification =====

ARCHIVE_REF_PATTERNS = [
    re.compile(r"\barchive/tests\b"),
    re.compile(r"\btests/.+_OLD\b"),
    re.compile(r"from\s+archive\.tests"),
    re.compile(r"import\s+archive\.tests"),
]


@cli_command(
    "verify-archive",
    help="Verify no references to archived tests",
    args={
        "fail_on_reference": {
            "type": bool,
            "action": "store_true",
            "help": "Exit with code 1 if references found",
        },
    },
)
def verify_archive(fail_on_reference: bool = False) -> int:
    """Check for textual references to archived tests or _OLD paths.

    Args:
        fail_on_reference: If True, exit with code 1 when references found

    Returns:
        0 if no references found, 1 if references found
    """
    matches = []
    for p in ROOT.rglob("**/*.*"):
        if not p.is_file():
            continue
        if any(part.startswith('.git') for part in p.parts):
            continue
        try:
            text = p.read_text(errors='ignore')
        except Exception:
            continue
        for pat in ARCHIVE_REF_PATTERNS:
            for m in pat.finditer(text):
                start = max(m.start() - 40, 0)
                end = min(m.end() + 40, len(text))
                snippet = text[start:end].replace('\n', ' ')
                matches.append((str(p.relative_to(ROOT)), m.group(0), snippet))

    if matches:
        print("Found references to archived tests or _OLD files:")
        for fname, token, snippet in matches:
            print(f" - {fname}: '{token}' -> ...{snippet}...")
        print('\nPlease review and remove references before moving tests to archive or update the reference target.')
        if fail_on_reference:
            return 1
    else:
        print("No references to archived tests found.")

    return 0


# ===== Archive Management =====

@cli_command(
    "archive-outputs",
    help="Archive outputs directory",
    args={
        "outputs": {
            "type": str,
            "required": True,
            "help": "Directory to archive",
        },
        "dest": {
            "type": str,
            "default": "data/archive",
            "help": "Destination directory",
        },
        "exclude": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Glob patterns to exclude",
        },
    },
)
def archive_outputs(
    outputs: str,
    dest: str = "data/archive",
    exclude: list[str] | None = None
) -> Path:
    """Archive outputs directory to compressed tarball.

    Args:
        outputs: Directory to archive
        dest: Destination directory for archive
        exclude: Glob patterns to exclude (relative to outputs_dir)

    Returns:
        Path to created archive
    """
    outputs_dir = Path(outputs)
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_name = f"outputs_archive_{timestamp}.tar.gz"
    archive_path = dest_dir / archive_name

    exclude = exclude or []

    with tarfile.open(archive_path, "w:gz") as tar:
        for root, dirs, files in __import__("os").walk(outputs_dir):
            root_path = Path(root)
            for fname in files:
                fpath = root_path / fname
                rel = fpath.relative_to(outputs_dir).as_posix()
                # check excludes
                skip = False
                for pat in exclude:
                    from fnmatch import fnmatch
                    if fnmatch(rel, pat):
                        skip = True
                        break
                if skip:
                    print(f"Excluding: {rel}")
                    continue
                tar.add(fpath, arcname=(outputs_dir.name + "/" + rel))

    print(f"Created archive: {archive_path}")
    return archive_path


def restore_archive(archive_path: Path, dest: Path) -> None:
    """Restore archive to destination directory.

    Args:
        archive_path: Path to .tar.gz archive
        dest: Destination directory
    """
    archive_path = Path(archive_path)
    dest = Path(dest)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest)
    print(f"Restored archive to: {dest}")


@cli_command(
    "list-archives",
    help="List available archives",
    args={
        "dir": {
            "type": str,
            "default": "data/archive",
            "help": "Directory containing archives",
        },
    },
)
def list_archives(dir: str = "data/archive") -> int:
    """List available archives in directory.

    Args:
        dir: Directory containing archives

    Returns:
        0 (exit code)
    """
    dir_path = Path(dir)
    if not dir_path.exists():
        print(f"Archive directory not found: {dir_path}")
        return 0
    
    archives = sorted(
        [p for p in dir_path.iterdir() if p.suffix in (".gz", ".tar")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not archives:
        print(f"No archives found in {dir_path}")
    else:
        print(f"Archives in {dir_path}:")
        for arch in archives:
            size = arch.stat().st_size / (1024 * 1024)  # MB
            mtime = datetime.fromtimestamp(arch.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {arch.name:50s} {size:8.1f} MB  {mtime}")
    
    return 0


# ===== Workspace Archiving =====

ARCHIVE_DIR = ROOT / "archive_local"

# Whitelist: NEVER archive these
WHITELIST_PATTERNS = {
    ".git",
    ".github",
    "data/images",
    "data/new_all_tiles.csv",
    "config/pipeline_config.yaml",
    "requirements.txt",
    "requirements-cpu.txt",
    "README.md",
    "pyproject.toml",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "archive_local",
    "scripts/deprecated",
    "outputs/final_selection",
}


def is_whitelisted(path: Path) -> bool:
    """Check if path matches whitelist patterns."""
    rel_path = path.relative_to(ROOT)
    path_str = str(rel_path)

    for pattern in WHITELIST_PATTERNS:
        if path_str.startswith(pattern) or pattern in path_str:
            return True
    return False


def get_file_age_days(path: Path) -> int:
    """Get file age in days."""
    if not path.exists():
        return 0
    mtime = path.stat().st_mtime
    age_seconds = datetime.now().timestamp() - mtime
    return int(age_seconds / 86400)


def archive_workspace(
    category: str = "all",
    dry_run: bool = True,
    age_threshold: int = 30
) -> int:
    """Archive old workspace files.

    Args:
        category: Category to archive (scripts/outputs/temp/all)
        dry_run: If True, only show what would be archived
        age_threshold: Minimum age in days to archive

    Returns:
        0 on success
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dest = ARCHIVE_DIR / f"workspace_cleanup_{timestamp}"

    candidates = []

    # Scan for candidates based on category
    if category in ("scripts", "all"):
        scripts_dir = ROOT / "scripts"
        if scripts_dir.exists():
            for script in scripts_dir.glob("*.py"):
                if not is_whitelisted(script) and get_file_age_days(script) > age_threshold:
                    candidates.append(script)

    if category in ("outputs", "all"):
        outputs_dir = ROOT / "outputs"
        if outputs_dir.exists():
            for output in outputs_dir.rglob("*"):
                if output.is_file() and not is_whitelisted(output) and get_file_age_days(output) > age_threshold:
                    candidates.append(output)

    if category in ("temp", "all"):
        for temp_pattern in ["*.tmp", "tmp_*", "temp_*"]:
            for temp_file in ROOT.glob(temp_pattern):
                if not is_whitelisted(temp_file):
                    candidates.append(temp_file)

    if not candidates:
        print(f"No files found for archiving (category={category}, age>{age_threshold}d)")
        return 0

    print(f"Found {len(candidates)} files to archive:")
    for f in candidates[:10]:  # Show first 10
        print(f"  - {f.relative_to(ROOT)}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")

    if dry_run:
        print("\n[DRY RUN] No files were archived. Use --yes to proceed.")
        return 0

    archive_dest.mkdir(parents=True, exist_ok=True)
    for f in candidates:
        rel_path = f.relative_to(ROOT)
        dest = archive_dest / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    print(f"\nArchived {len(candidates)} files to: {archive_dest}")
    return 0
