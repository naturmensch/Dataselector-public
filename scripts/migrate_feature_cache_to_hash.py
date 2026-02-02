#!/usr/bin/env python3
"""Migration script to move legacy outputs/features.npy -> features-{meta_hash}.npy

"""
import argparse
import shutil
import sys
from pathlib import Path

from src.cache import compute_meta_hash, atomic_write_features_with_meta, create_meta_info
import numpy as np


def migrate(out_dir: Path, csv_meta: Path, dry_run: bool = False) -> int:
    legacy = out_dir / "features.npy"
    if not legacy.exists():
        print("No legacy features.npy found; nothing to do.")
        return 0

    try:
        feats = np.load(legacy)
    except Exception as exc:
        print(f"ERROR: Could not read legacy features.npy: {exc}")
        return 2

    try:
        meta_hash = compute_meta_hash(str(csv_meta), params={})
    except Exception as exc:
        print(f"ERROR: Could not compute metadata hash: {exc}")
        return 2

    meta_info = create_meta_info(str(csv_meta), params={})
    target = out_dir / f"features-{meta_hash}.npy"

    if target.exists():
        print(f"Target cache {target} already exists; creating a backup of legacy and leaving it in place.")
        backup_dir = out_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        ts = backup_dir / f"features_legacy_backup_{int(__import__('time').time())}.npy"
        if not dry_run:
            shutil.move(str(legacy), str(ts))
        print(f"Legacy moved to {ts}")
        return 0

    if dry_run:
        print(f"Would write {target} with meta hash {meta_hash}")
        return 0

    # Perform atomic write and backup
    atomic_write_features_with_meta(out_dir, feats, meta_hash, meta_info)
    backup_dir = out_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = backup_dir / f"features_legacy_backup_{int(__import__('time').time())}.npy"
    shutil.move(str(legacy), str(ts))
    print(f"Migrated legacy features.npy -> {target} and backed up original to {ts}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--csv", default="data/new_all_tiles.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(migrate(Path(args.out_dir), Path(args.csv), dry_run=args.dry_run))
