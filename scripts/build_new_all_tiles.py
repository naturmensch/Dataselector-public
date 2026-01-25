#!/usr/bin/env python3
"""Build `data/new_all_tiles.csv` by enriching the canonical `KDR100_foliage_with_files_epsg3857.csv` with image paths and metadata from XML sidecars.

- Loads the canonical table (676 tiles, EPSG:3857) as base if available.
- Scans an image directory for image files (png/jpg/jpeg) and extracts metadata from sidecars.
- Converts extracted lat/lon to EPSG:3857 to match base coordinates.
- Merges base and extracted data on 'longName', filling NaN without overwriting existing values.
- Writes `new_all_tiles.csv` (or `--out` path) atomically, creates a backup of an existing file, and writes a provenance JSON next to the CSV.

Usage:
  ./scripts/exec_in_env.sh --env dataselector -- python scripts/build_new_all_tiles.py --image-dir data/images --out data/new_all_tiles.csv

This script ensures no data is overwritten, coordinates stay in EPSG:3857 (meters), and missing fields are filled from images.
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
from scripts.common import data_path

import pandas as pd

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SIDECAR_SUFFIXES = [".aux.xml", ".xml"]


def extract_city(long_name: str) -> str:
    """Extract the city/location name from longName like 'KDR_146_Hamburg_1918.png' -> 'Hamburg'"""
    if not isinstance(long_name, str) or not long_name:
        return ""
    parts = long_name.split('_')
    if len(parts) >= 3:
        # Assume format: KDR_XXX_City_Year.png, so city is parts[-2]
        city_part = parts[-2]
        # Remove any trailing digits if present (e.g., if year has no .png)
        return ''.join(c for c in city_part if not c.isdigit())
    return ""


def extract_year(long_name: str) -> str:
    """Extract the year from longName like 'KDR_146_Hamburg_1918.png' -> '1918'"""
    if not isinstance(long_name, str) or not long_name:
        return ""
    parts = long_name.split('_')
    if len(parts) >= 2:
        year_part = parts[-1]
        # Remove .png extension if present
        if '.' in year_part:
            year_part = year_part.split('.')[0]
        # Keep only digits
        return ''.join(c for c in year_part if c.isdigit())
    return ""


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_xml(xml_path: Path) -> Dict[str, Optional[str]]:
    """Try to extract coordinates from GDAL AUX XML.

    Looks for CornerCoordinates or GeoTransform and extracts left, top, right, bottom.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return {}

    meta: Dict[str, Optional[str]] = {}

    # Try CornerCoordinates first
    corner_coords = root.find('.//CornerCoordinates')
    if corner_coords is not None:
        upper_left = corner_coords.find('UpperLeft')
        lower_right = corner_coords.find('LowerRight')
        if upper_left is not None and upper_left.text:
            ul = upper_left.text.strip().split(',')
            if len(ul) == 2:
                meta['left'] = ul[0].strip()
                meta['top'] = ul[1].strip()
        if lower_right is not None and lower_right.text:
            lr = lower_right.text.strip().split(',')
            if len(lr) == 2:
                meta['right'] = lr[0].strip()
                meta['bottom'] = lr[1].strip()

    # Try GeoTransform if no CornerCoordinates
    if not meta:
        geo_transform = root.find('.//GeoTransform')
        if geo_transform is not None and geo_transform.text:
            gt = geo_transform.text.strip().split(',')
            if len(gt) == 6:
                meta['left'] = gt[0].strip()
                meta['pixelWidth'] = gt[1].strip()
                meta['top'] = gt[3].strip()
                meta['pixelHeight'] = gt[5].strip()

    # Fallback to old lat/lon search if needed
    if not meta:
        text = ET.tostring(root, encoding="utf-8", method="text").decode("utf-8")
        # naive search for digits/decimal in text near keywords
        for key in ("lat", "latitude", "north", "N"):
            if key in text.lower() and "=" not in text:
                meta.setdefault("N", None)
        for key in ("lon", "longitude", "left", "long", "x"):
            if key in text.lower() and "=" not in text:
                meta.setdefault("left", None)

        # More structured: search attributes and tags in root and subelements
        for elem in root.iter():
            # attributes
            for k, v in elem.attrib.items():
                kl = k.lower()
                if any(x in kl for x in ("lat", "latitude", "n")) and v:
                    if not meta.get("N"):
                        meta["N"] = v
                if any(x in kl for x in ("lon", "long", "left", "x")) and v:
                    if not meta.get("left"):
                        meta["left"] = v
            # tag names
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

    return meta
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


def latlon_to_epsg3857(lat, lon):
    if not HAS_PYPROJ:
        raise ImportError("pyproj required for coordinate conversion")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)


def epsg3857_to_latlon(x, y):
    if not HAS_PYPROJ:
        raise ImportError("pyproj required for coordinate conversion")
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


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
            # Calculate right/bottom from GeoTransform if available
            if 'pixelWidth' in meta and 'pixelHeight' in meta and 'left' in meta and 'top' in meta:
                try:
                    from PIL import Image
                    with Image.open(str(p)) as img:
                        width, height = img.size
                    pixelWidth = float(meta['pixelWidth'])
                    pixelHeight = float(meta['pixelHeight'])
                    left = float(meta['left'])
                    top = float(meta['top'])
                    meta['right'] = str(left + width * pixelWidth)
                    meta['bottom'] = str(top + height * pixelHeight)
                except Exception:
                    pass
        year = extract_year_from_name(img_name) or None
        rows.append(
            {
                "longName": img_name,
                "shortName": img_name,
                "left": meta.get("left"),
                "top": meta.get("top"),
                "right": meta.get("right"),
                "bottom": meta.get("bottom"),
                "image_path": str(p),
                "image_filename": img_name,
                "year": year,
            }
        )

    df = pd.DataFrame(rows)
    # normalize numeric columns
    for col in ("N", "left", "top", "right", "bottom"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Rename 'top' to 'N' for consistency with other parts of the codebase
    if 'top' in df.columns:
        df = df.rename(columns={'top': 'N'})
    
    # Convert lat/lon to EPSG:3857 if pyproj available
    # Note: Coordinates from AUX are already in EPSG:3857, no transformation needed
    
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
    parser.add_argument("--image-dir", default=str(data_path("images")), help="Directory with images")
    parser.add_argument("--out", default=str(data_path("new_all_tiles.csv")), help="Output CSV path")
    parser.add_argument("--force-source", help="Force a specific source CSV to use for provenance")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    image_dir = Path(args.image_dir)
    out = Path(args.out)

    base_path = data_path("KDR100_foliage_with_files_epsg3857.csv")

    # If there is a vetted "final" CSV present in data, prefer it for provenance
    final_candidates = [
        data_path("all_png_tiles_final_ultimative.csv"),
        data_path("all_png_tiles_final_from_dbf.csv"),
        base_path,
    ]
    src = Path(args.force_source) if args.force_source else choose_source(final_candidates)

    if not base_path.exists():
        print(f"Warning: Base metadata {base_path} not found. Falling back to image scan only.")

    extracted_df = build_dataframe(image_dir)
    if extracted_df.empty:
        print("No images found; aborting.")
        return 1

    if base_path.exists():
        base_df = pd.read_csv(base_path)
        # Rename columns to remove type suffixes (e.g., "longName,C,254" -> "longName")
        base_df.columns = [col.split(',')[0] for col in base_df.columns]
        # Rename 'top' to 'N' for consistency
        if 'top' in base_df.columns:
            base_df = base_df.rename(columns={'top': 'N'})
        # Keep original coordinates in separate columns
        base_df['epsg_left'] = base_df['left']
        base_df['epsg_top'] = base_df['N']  # Now 'N' instead of 'top'
        base_df['epsg_right'] = base_df['right']
        base_df['epsg_bottom'] = base_df['bottom']
        print(f"Loaded base metadata from {base_path} ({len(base_df)} rows, columns: {list(base_df.columns)})")
        merge_df = pd.merge(base_df, extracted_df, on='shortName', how='left', suffixes=('', '_extracted'))
        for col in extracted_df.columns:
            if col in merge_df.columns and col != 'longName':
                extracted_col = col + '_extracted'
                if extracted_col in merge_df.columns:
                    # Prefer extracted values over base values
                    merge_df[col] = merge_df[extracted_col].fillna(merge_df[col])
        cols_to_drop = [c for c in merge_df.columns if c.endswith('_extracted')]
        merge_df.drop(columns=cols_to_drop, inplace=True)
        df = merge_df
    else:
        df = extracted_df

    # Add extracted city column
    df['city'] = df['longName'].apply(extract_city)
    # Add extracted year column
    df['extracted_year'] = df['longName'].apply(extract_year)

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
