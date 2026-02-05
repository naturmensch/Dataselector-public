"""Documentation link maintenance tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from dataselector.cli_decorators import cli_command

ROOT = Path(__file__).resolve().parents[2]  # dataselector/tools/docs_link.py -> repo root


def find_broken_links(docs_dir: Path = None) -> List[Tuple[Path, str, str]]:
    """Find broken relative links in markdown files.

    Args:
        docs_dir: Directory to scan (defaults to ROOT/docs)

    Returns:
        List of (source_file, link_target, link_text) tuples for broken links
    """
    docs_dir = docs_dir or ROOT / "docs"
    if not docs_dir.exists():
        return []

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    broken = []

    for md_file in docs_dir.rglob("*.md"):
        text = md_file.read_text(encoding='utf-8', errors='ignore')
        for match in link_pattern.finditer(text):
            link_text = match.group(1)
            link_target = match.group(2)

            # Skip external links
            if link_target.startswith(('http://', 'https://', 'mailto:', '#')):
                continue

            # Resolve relative path
            target_path = (md_file.parent / link_target).resolve()
            if not target_path.exists():
                broken.append((md_file, link_target, link_text))

    return broken


@cli_command(
    "docs-link-autofix",
    help="Auto-fix broken documentation links",
    args={
        "yes": {
            "type": bool,
            "action": "store_true",
            "help": "Actually fix links (default is dry-run)",
        },
        "no_backup": {
            "type": bool,
            "action": "store_true",
            "help": "Don't backup original files",
        },
    },
)
def autofix_links(yes: bool = False, no_backup: bool = False) -> int:
    """Attempt to auto-fix broken relative links.

    Args:
        yes: If True, actually fix (default is dry-run)
        no_backup: If True, don't backup original files

    Returns:
        0 on success
    """
    dry_run = not yes  # Invert: --yes means NOT dry-run
    backup = not no_backup  # Invert: --no-backup means NO backup
    broken = find_broken_links()

    if not broken:
        print("No broken links found.")
        return 0

    print(f"Found {len(broken)} broken links")

    archive_dir = ROOT / 'archive_local' / 'docs_migration_backup' / 'autofix'
    if backup and not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    fixed = []
    manual = []

    for src_file, target, text in broken:
        # Try to find target by basename
        target_basename = Path(target).name
        candidates = [
            p for p in ROOT.rglob('*')
            if p.is_file() and p.name.lower() == target_basename.lower()
        ]

        if len(candidates) == 1:
            # Unique match found
            cand = candidates[0]
            import os
            rel_path = os.path.relpath(str(cand), start=str(src_file.parent))

            if dry_run:
                print(f"Would fix: {src_file.relative_to(ROOT)}")
                print(f"  {target} -> {rel_path}")
            else:
                # Backup original
                if backup:
                    backup_path = archive_dir / (src_file.name + '.orig')
                    if not backup_path.exists():
                        backup_path.write_bytes(src_file.read_bytes())

                # Replace in file
                content = src_file.read_text(encoding='utf-8')
                new_content = content.replace(f"]({target})", f"]({rel_path})")
                src_file.write_text(new_content, encoding='utf-8')
                fixed.append((src_file, target, rel_path))
        else:
            manual.append((src_file, target, len(candidates)))

    if dry_run:
        print(f"\n[DRY RUN] Would fix {len(fixed)} links automatically")
        print(f"Would require manual fix: {len(manual)} links")
    else:
        print(f"\nFixed {len(fixed)} links automatically")
        print(f"Require manual fix: {len(manual)} links")

        for src, tgt, n_cand in manual:
            print(f"  - {src.relative_to(ROOT)}: {tgt} ({n_cand} candidates)")

    return 0


@cli_command(
    "docs-link-check",
    help="Check for broken documentation links",
    args={},
)
def check_links() -> int:
    """Check for broken links in documentation.

    Returns:
        0 if no broken links, 1 if broken links found
    """
    broken = find_broken_links()

    if not broken:
        print("✓ No broken links found")
        return 0

    print(f"✗ Found {len(broken)} broken links:")
    for src_file, target, text in broken:
        print(f"  - {src_file.relative_to(ROOT)}: [{text}]({target})")

    return 1
