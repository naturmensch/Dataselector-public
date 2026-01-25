#!/usr/bin/env python3
"""Hilfs-Skript: Zählt CSV-Zeilen in einem Archiv (ohne Header).

Usage:
  python scripts/count_archived_tiles.py --archive data/archive/archive-20260111-234742.tar.gz
  python scripts/count_archived_tiles.py --archive <path> --extract <member> --out data/

Outputs a simple report to stdout and returns exit code 0 on success.
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path
from typing import Dict


def count_csv_lines_in_tar(tar_path: str | Path) -> Dict[str, int]:
    tar_path = Path(tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    counts: Dict[str, int] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".csv"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                # Read text bytes safely and count lines
                try:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    lines = sum(1 for _ in text)
                finally:
                    try:
                        text.close()
                    except Exception:
                        pass
                # Assume first line is header if >1 line
                count = lines - 1 if lines > 0 else 0
                counts[member.name] = count
    return counts


def extract_member(tar_path: str | Path, member_name: str, out_dir: str | Path) -> Path:
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name == member_name), None)
        if member is None:
            raise ValueError(f"Member not found in archive: {member_name}")
        tar.extract(member, path=out_dir)
        return out_dir / member.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, required=True)
    parser.add_argument("--extract", type=str, default=None)
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    try:
        counts = count_csv_lines_in_tar(args.archive)
    except Exception as e:
        print(f"Error: {e}")
        return 2

    if not counts:
        print("No CSV files found in archive.")
        return 0

    print("CSV files found in archive and their row counts (header excluded):")
    for name, c in counts.items():
        print(f"- {name}: {c}")

    if args.extract:
        try:
            dest = extract_member(args.archive, args.extract, args.out)
            print(f"Extracted {args.extract} -> {dest}")
        except Exception as e:
            print(f"Could not extract {args.extract}: {e}")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
