#!/usr/bin/env python3
"""Safety-first cleanup helpers for the workspace.

Usage:
  python scripts/clean_workspace.py --dry-run
  python scripts/clean_workspace.py --delete-outputs --delete-venvs
  python scripts/clean_workspace.py --archive data/images /path/to/archive.tar.gz

The script is intentionally conservative: nothing is deleted unless explicit flags are given.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import tarfile
import sys
import os

PROTECTED = set([
    'data/images',
    'data/archive',
    'data/raw',
    'models',
    'outputs/final_selection',
    'outputs/kdr100_selection'
])

# Allow additional protected paths via environment variable PROTECTED_PATHS
# (comma-separated) or via CLI `--protect` option
CANDIDATES = {
    'outputs/validation': 'outputs/validation',
    'outputs/cache_backup': 'outputs/cache_backup_20260112',
    'data/images': 'data/images',  # PROTECTED: will never be deleted by default
    'data/archive': 'data/archive',
    'data/raw': 'data/raw',
    'models': 'models',
    'outputs/final_selection': 'outputs/final_selection',
    'outputs/kdr100_selection': 'outputs/kdr100_selection',
    '.venv': '.venv',
    'venv': 'venv',
}


def sizeof_fmt(num: int, suffix='B') -> str:
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def archive_path(path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(path, arcname=path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Only list candidates and sizes')
    parser.add_argument('--delete-outputs', action='store_true', help='Delete known output directories')
    parser.add_argument('--delete-venvs', action='store_true', help='Delete local virtual environment dirs (.venv, venv)')
    parser.add_argument('--archive', nargs=2, metavar=('SRC','ARCHIVE'), help='Archive SRC to ARCHIVE (tar.gz)')
    parser.add_argument('--protect', action='append', help='Add additional protected path (repeatable)')
    args = parser.parse_args()

    # Merge PROTECTED with environment variable and CLI --protect
    env = os.environ.get('PROTECTED_PATHS')
    if env:
        for p in env.split(','):
            p = p.strip()
            if p:
                PROTECTED.add(p)
    if args.protect:
        for p in args.protect:
            PROTECTED.add(p)

    print('Scanning workspace for large candidates...')
    found = []
    for key, rel in CANDIDATES.items():
        p = Path(rel)
        if p.exists():
            size = dir_size(p) if p.is_dir() else p.stat().st_size
            prot = ' (PROTECTED — will not be deleted)' if rel in PROTECTED else ''
            print(f" - {p}: {sizeof_fmt(size)}{prot}")
            found.append((p, size, rel in PROTECTED))

    if not found:
        print('No candidate directories found.')
        return

    if args.dry_run:
        print('\nDry-run: no changes made. Use --delete-outputs or --delete-venvs to remove items, or --archive SRC ARCHIVE to archive.')
        return

    if args.archive:
        src, arch = args.archive
        src_p = Path(src)
        arch_p = Path(arch)
        if not src_p.exists():
            print(f"Source {src} does not exist")
            sys.exit(1)
        print(f"Archiving {src} -> {arch}")
        archive_path(src_p, arch_p)
        print('Archive complete')

    if args.delete_outputs:
        for p, _, is_prot in found:
            if is_prot:
                print(f"Skipping protected {p}")
                continue
            if str(p).startswith('outputs') and p.exists():
                print(f"Deleting {p}")
                shutil.rmtree(p)

    if args.delete_venvs:
        for p, _, is_prot in found:
            if is_prot:
                continue
            if p.name in ('.venv','venv') and p.exists():
                print(f"Deleting {p}")
                shutil.rmtree(p)


if __name__ == '__main__':
    main()
