#!/usr/bin/env python3
"""Install git hooks from `.githooks/` into `.git/hooks/`.

Run this only if you want to enable project-local hooks:

  python scripts/install_git_hooks.py

The script is conservative and will not overwrite existing hooks unless --force is passed.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HOOKS_DIR = Path(".githooks")
GIT_HOOKS = Path(".git") / "hooks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing hooks")
    args = parser.parse_args()

    if not HOOKS_DIR.exists():
        print(f"No {HOOKS_DIR} directory found; nothing to install")
        return 0
    if not GIT_HOOKS.exists():
        print("No .git/hooks directory found — are you in a git repository?")
        return 1

    for hook in HOOKS_DIR.iterdir():
        dest = GIT_HOOKS / hook.name
        if dest.exists() and not args.force:
            print(f"Skipping existing hook {dest}; use --force to overwrite")
            continue
        shutil.copy(hook, dest)
        dest.chmod(0o755)
        print(f"Installed {hook.name} -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
