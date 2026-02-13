#!/usr/bin/env python3
"""
Copy selected tiles and their .png.aux.xml sidecars to a target directory.

Usage:
  python scripts/copy_selection_tiles.py --csv <selection.csv> [--outdir <path>] [--overwrite] [--dry-run]
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
import sys


def copy_selected(csv_path: Path, outdir: Path, overwrite: bool, dry_run: bool) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "selection_manifest.csv"
    rows_written = 0

    with csv_path.open(newline="", encoding="utf-8") as fh_in, manifest_path.open("w", newline="", encoding="utf-8") as fh_out:
        reader = csv.DictReader(fh_in)
        fieldnames = ["shortName", "image_filename", "src_image", "dst_image", "sidecar_src", "sidecar_dst", "sidecar_copied"]
        writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
        writer.writeheader()

        for r in reader:
            img_rel = r.get("image_path") or r.get("image_filename") or r.get("filename")
            short = r.get("shortName") or r.get("shortname") or ""
            if not img_rel:
                print(f"[WARN] row missing image path / filename: {r}", file=sys.stderr)
                continue

            src_img = Path(img_rel)
            if not src_img.is_absolute():
                src_img = Path.cwd() / src_img

            dst_img = outdir / src_img.name
            sidecar_src = src_img.with_name(src_img.name + ".aux.xml")
            sidecar_dst = outdir / sidecar_src.name

            for src, dst, want_copy in (
                (src_img, dst_img, src_img.exists()),
                (sidecar_src, sidecar_dst, sidecar_src.exists()),
            ):
                if not want_copy:
                    continue

                if dst.exists() and not overwrite:
                    print(f"[INFO] exists, skipping (use --overwrite to replace): {dst}")
                else:
                    action = "DRY-RUN copy" if dry_run else "copy"
                    print(f"[{action}] {src} -> {dst}")
                    if not dry_run:
                        shutil.copy2(src, dst)

            writer.writerow({
                "shortName": short,
                "image_filename": src_img.name,
                "src_image": str(src_img),
                "dst_image": str(dst_img),
                "sidecar_src": str(sidecar_src) if sidecar_src.exists() else "",
                "sidecar_dst": str(sidecar_dst) if sidecar_src.exists() else "",
                "sidecar_copied": "yes" if sidecar_src.exists() else "no",
            })
            rows_written += 1

    print(f"[✓] Manifest written: {manifest_path} ({rows_written} rows)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path, help="Selection CSV (exported by pipeline)")
    p.add_argument("--outdir", type=Path, default=Path("<qgis-data-root>"), help="Destination folder")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="Show actions without copying")
    p.add_argument("--flat", action="store_true", help="Do not create a subfolder inside --outdir; copy files directly into outdir")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    # by default create a subfolder inside outdir named after the selection CSV (helps keep runs isolated)
    dest_dir = args.outdir if args.flat else (args.outdir / args.csv.stem)

    sys.exit(copy_selected(args.csv, dest_dir, args.overwrite, args.dry_run))


if __name__ == "__main__":
    main()
