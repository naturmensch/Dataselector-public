#!/usr/bin/env python3
"""Simple archive management for outputs.

Usage:
  scripts/manage_archives.py archive --outputs outputs --dest data/archive
  scripts/manage_archives.py restore --archive data/archive/outputs_archive_20260112_152121.tar.gz --dest .
  scripts/manage_archives.py list --dir data/archive

This is intentionally small and dependency-free (only stdlib).
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tarfile
import sys


def archive_outputs(outputs_dir: Path, dest_dir: Path, prefix: str = "outputs_archive") -> Path:
    outputs_dir = Path(outputs_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{prefix}_{timestamp}.tar.gz"
    archive_path = dest_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(outputs_dir, arcname=outputs_dir.name)

    return archive_path


def restore_archive(archive_path: Path, dest: Path) -> None:
    archive_path = Path(archive_path)
    dest = Path(dest)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest)


def list_archives(dir_path: Path):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.suffix in (".gz", ".tar")], reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage_archives.py")
    sub = parser.add_subparsers(dest="cmd")

    p_archive = sub.add_parser("archive")
    p_archive.add_argument("--outputs", required=True)
    p_archive.add_argument("--dest", default="data/archive")

    p_restore = sub.add_parser("restore")
    p_restore.add_argument("--archive", required=True)
    p_restore.add_argument("--dest", default=".")

    p_list = sub.add_parser("list")
    p_list.add_argument("--dir", default="data/archive")

    args = parser.parse_args(argv)

    if args.cmd == "archive":
        archive_path = archive_outputs(Path(args.outputs), Path(args.dest))
        print(f"Created archive: {archive_path}")
        return 0
    if args.cmd == "restore":
        restore_archive(Path(args.archive), Path(args.dest))
        print(f"Restored archive: {args.archive} -> {args.dest}")
        return 0
    if args.cmd == "list":
        archives = list_archives(Path(args.dir))
        for a in archives:
            print(a)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
