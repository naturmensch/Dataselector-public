#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

# Default candidates shown in dry-run; tests may monkeypatch this mapping.
CANDIDATES: Dict[str, str] = {
    "outputs/final_selection": "outputs/final_selection",
    "data/images": "data/images",
    "data/archive": "data/archive",
}


def _is_outputs_path(rel: str) -> bool:
    return rel == "outputs" or rel.startswith("outputs/")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Workspace cleanup utility (safe defaults)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument(
        "--delete-outputs",
        action="store_true",
        help="Remove entries under outputs/ listed in CANDIDATES",
    )
    args = parser.parse_args(argv)

    # Always print status for each candidate
    for rel in CANDIDATES.keys():
        status = "PROTECTED"
        if args.delete_outputs and _is_outputs_path(rel):
            status = "DELETE"
        print(f"{rel}: {status}")

    if args.dry_run:
        return 0

    if args.delete_outputs:
        for rel in list(CANDIDATES.keys()):
            if not _is_outputs_path(rel):
                continue
            ap = Path(rel)
            if ap.exists():
                try:
                    if ap.is_file():
                        ap.unlink()
                    else:
                        shutil.rmtree(ap)
                    print(f"Removed: {ap}")
                except Exception as e:
                    print(f"Warning: failed to remove {ap}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
