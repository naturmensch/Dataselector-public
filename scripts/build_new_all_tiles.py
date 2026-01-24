#!/usr/bin/env python3
"""Build minimal `data/new_all_tiles.csv` from raw image files and XML sidecars.

- Scans an image directory for image files (png/jpg/jpeg)
- For each image, tries to find a sidecar with suffixes (".aux.xml", ".xml")
  and parse basic metadata (lat / lon) if present.
- Writes `new_all_tiles.csv` (or `--out` path) atomically, creates a backup of
  an existing file, and writes a provenance JSON next to the CSV.

Usage:
  python scripts/build_new_all_tiles.py --image-dir data/images --out data/new_all_tiles.csv

This script is intentionally small and robust: missing XML fields are allowed,
but are logged and left blank (NaN in CSV).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SIDECAR_SUFFIXES = [".aux.xml", ".xml"]


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_xml(xml_path: Path) -> Dict[str, Optional[str]]:
    """Try to extract lat / lon or geotransform-like values from XML.

    This is a best-effort parser that looks for common tags/attributes and
    returns a dict with keys 'N' and 'left' (as strings) when available.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return {}

    meta: Dict[str, Optional[str]] = {}

    # Common patterns: elements containing 'lat' / 'lon' or attributes
    text = ET.tostring(root, encoding="utf-8", method="text").decode("utf-8")
    # naive search for digits/decimal in text near keywords
    for key in ("lat", "latitude", "north", "N"):
        if key in text.lower() and "=" not in text:
            # attempt manual find
            meta.setdefault("N", None)
    for key in ("lon", "longitude", "left", "long", "x"):
        if key in text.lower() and "=" not in text:
            meta.setdefault("left", None)

    # More structured: search attributes and tags in root and subelements
    for elem in root.iter():
        # attributes (prefer attributes but do not overwrite later)
        for k, v in elem.attrib.items():
            kl = k.lower()
            if any(x in kl for x in ("lat", "latitude", "n")) and v:
                if not meta.get("N"):
                    meta["N"] = v
            if any(x in kl for x in ("lon", "long", "left", "x")) and v:
                if not meta.get("left"):
                    meta["left"] = v
        # tag names (prefer explicit names lat/lon)
        tagname = elem.tag
        if isinstance(tagname, str) and '}' in tagname:
            tagname = tagname.split('}', 1)[1]
        ltag = str(tagname).lower()
        txt = elem.text.strip() if elem.text and isinstance(elem.text, str) else None
        if txt:
            if "lat" in ltag or ltag in ("n", "north"):
                if not meta.get("N"):
                    meta["N"] = txt
            elif "lon" in ltag or "left" in ltag or "long" in ltag or "x" in ltag:
                if not meta.get("left"):
                    meta["left"] = txt

    # Normalize simple numeric strings
    for k in ("N", "left"):
        if k in meta and isinstance(meta[k], str):
            s = meta[k].strip()
            s = s.replace(",", ".")
            # keep as string; conversion to numeric can happen downstream
            meta[k] = s

    return meta


def extract_year_from_name(name: str) -> Optional[int]:
    import re

    m = re.search(r"(17|18|19|20)\d{2}", name)
    if m:
        return int(m.group(0))
    return None


def build_dataframe(image_dir: Path) -> pd.DataFrame:
    rows = []
    if not image_dir.exists():
        raise SystemExit(f"Image dir not found: {image_dir}")

    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        img_name = p.name
        stem = p.stem
        # find sidecar
        side = None
        for suf in SIDECAR_SUFFIXES:
            candidate = p.with_suffix(p.suffix + suf) if suf.startswith(".") and not str(p).endswith(suf) else p.with_suffix(suf)
            # fallback search by name + suffix
            if candidate.exists():
                side = candidate
                break
        # fallback: check <stem> + suffix
        if side is None:
            for suf in SIDECAR_SUFFIXES:
                candidate = image_dir / f"{stem}{suf}"
                if candidate.exists():
                    side = candidate
                    break
        meta: Dict[str, Optional[str]] = {}
        if side is not None:
            meta = extract_from_xml(side)
        year = extract_year_from_name(img_name) or None
        rows.append(
            {
                "longName": img_name,
                "shortName": stem,
                "N": meta.get("N"),
                "left": meta.get("left"),
                "image_path": str(p),
                "image_filename": img_name,
                "year": year,
            }
        )

    df = pd.DataFrame(rows)
    # normalize numeric columns
    for col in ("N", "left"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def atomic_write_csv(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # backup
    if out.exists():
        bak = out.with_suffix(f".backup_{int(time.time())}.csv")
        shutil.copy2(out, bak)
    # write to temp and move
    fd, tmpname = tempfile.mkstemp(dir=str(out.parent), prefix=".new_all_tiles_")
    os.close(fd)
    df.to_csv(tmpname, index=False)
    os.replace(tmpname, out)


def write_provenance(src: Path, out: Path, rows: int) -> None:
    prov = {
        "generated_at": int(time.time()),
        "source": str(src.name),
        "source_sha256": sha256_of_file(src) if src.exists() else None,
        "rows": rows,
    }
    (out.parent / (out.stem + "_provenance.json")).write_text(json.dumps(prov, indent=2))


def choose_source(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/images", help="Directory with images")
    parser.add_argument("--out", default="data/new_all_tiles.csv", help="Output CSV path")
    parser.add_argument("--force-source", help="Force a specific source CSV to use for provenance")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    image_dir = Path(args.image_dir)
    out = Path(args.out)

    # If there is a vetted "final" CSV present in data, prefer it for provenance
    final_candidates = [
        root / "data" / "all_png_tiles_final_ultimative.csv",
        root / "data" / "all_png_tiles_final_from_dbf.csv",
        root / "data" / "KDR100_foliage_with_files_epsg3857.csv",
    ]
    src = Path(args.force_source) if args.force_source else choose_source(final_candidates)

    df = build_dataframe(image_dir)
    if df.empty:
        print("No images found; aborting.")
        return 1

    # Write CSV atomically and provenance
    atomic_write_csv(df, out)
    if src:
        write_provenance(src, out, len(df))
    else:
        # create minimal provenance indicating only image-dir
        prov = {
            "generated_at": int(time.time()),
            "source": "images_scanned",
            "image_dir": str(image_dir),
            "rows": len(df),
        }
        (out.parent / (out.stem + "_provenance.json")).write_text(json.dumps(prov, indent=2))

    print(f"Wrote {out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
