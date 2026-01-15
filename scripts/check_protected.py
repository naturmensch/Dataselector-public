#!/usr/bin/env python3
"""Check staged files for modifications inside protected paths.

Usage:
  python scripts/check_protected.py           # exits 0 if OK, non-zero if offending files
  python scripts/check_protected.py --list    # list protected paths
  python scripts/check_protected.py --protect data/secret --protect outputs/kdr100_selection
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_PROTECTED = {
    "data/images",
    "data/archive",
    "data/raw",
    "models",
    "outputs/final_selection",
    "outputs/kdr100_selection",
}
ENV_VAR = "PROTECTED_PATHS"  # comma-separated


def get_protected_paths(extra: Iterable[str] | None = None) -> Set[str]:
    s = set(DEFAULT_PROTECTED)
    env = os.environ.get(ENV_VAR)
    if env:
        for p in env.split(','):
            p = p.strip()
            if p:
                s.add(p)
    if extra:
        for p in extra:
            s.add(p)
    return s


def normalize(p: str) -> Path:
    return Path(p).as_posix()


def offending_files(staged_files: Iterable[str], protected_paths: Iterable[str]) -> List[str]:
    prot = [Path(p) for p in protected_paths]
    offenders: List[str] = []
    for f in staged_files:
        pf = Path(f)
        for p in prot:
            try:
                if p == pf or p in pf.parents:
                    offenders.append(f)
                    break
            except Exception:
                # defensive
                if str(f).startswith(str(p)):
                    offenders.append(f)
                    break
    return offenders


def git_staged_files() -> List[str]:
    # git diff --name-only --cached
    try:
        out = subprocess.check_output(["git", "diff", "--name-only", "--cached"], stderr=subprocess.DEVNULL)
        files = out.decode().splitlines()
        return [f for f in files if f]
    except Exception:
        # Not a git repo or git not available — return empty list
        return []


def git_tracked_files() -> List[str]:
    """Return a list of tracked files in the repository (git ls-files).

    Useful for CI checks that need to ensure protected files aren't tracked.
    """
    try:
        out = subprocess.check_output(["git", "ls-files"], stderr=subprocess.DEVNULL)
        files = out.decode().splitlines()
        return [f for f in files if f]
    except Exception:
        return []


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--list', action='store_true', help='List protected paths and exit')
    p.add_argument('--protect', action='append', help='Add protected path (repeatable)')
    p.add_argument('--staged', nargs='*', help='Provide staged files explicitly (for testing)')
    p.add_argument('--all', action='store_true', help='Check all tracked files (git ls-files)')
    args = p.parse_args(argv)

    protected = get_protected_paths(args.protect)

    if args.list:
        for x in sorted(protected):
            print(x)
        return 0

    staged = None
    if args.staged is not None:
        staged = args.staged
    elif args.all:
        staged = git_tracked_files()
    else:
        staged = git_staged_files()

    offenders = offending_files(staged, protected)
    if offenders:
        print("ERROR: The following staged files are inside protected paths:")
        for f in offenders:
            print(f"  {f}")
        print("\nPlease remove them from the commit or add an exception. See PROTECTED_PATHS env var to configure.")
        return 2

    # All good
    return 0


if __name__ == '__main__':
    raise SystemExit(main())