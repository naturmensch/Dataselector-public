#!/usr/bin/env python3
"""Build new_all_tiles.csv from image directory scan.

Scans image directory for .png/.jpg/.jpeg files, extracts metadata from sidecar
.aux.xml files, and writes atomic CSV with provenance JSON.

Usage:
    python -m dataselector.data.build_tiles --image-dir data/images --out data/new_all_tiles.csv
"""

from __future__ import annotations

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

from dataselector.cli_decorators import cli_command

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SIDECAR_SUFFIXES = [".aux.xml", ".xml"]


def sha256_of_file(p: Path) -> str:
    """Calculate SHA256 hash of file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_xml(xml_path: Path) -> Dict[str, Optional[str]]:
    """Extract metadata from XML sidecar file."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return {}

    meta: Dict[str, Optional[str]] = {}
    for child in root:
        if child.text:
            k = child.tag
            s = child.text.strip()
            if s:
                meta[k] = s

    return meta


def extract_year_from_name(name: str) -> Optional[int]:
    """Extract 4-digit year from filename."""
    import re

    m = re.search(r"(17|18|19|20)\d{2}", name)
    if m:
        return int(m.group(0))
    return None


def build_dataframe(image_dir: Path):
    """Scan image directory and build DataFrame with metadata."""
    import pandas as pd
    
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
        
        # Find sidecar XML
        side = None
        for suf in SIDECAR_SUFFIXES:
            candidate = p.with_suffix(p.suffix + suf) if suf.startswith(".") and not str(p).endswith(suf) else p.with_suffix(suf)
            if candidate.exists():
                side = candidate
                break
        
        # Fallback: check <stem> + suffix
        if side is None:
            for suf in SIDECAR_SUFFIXES:
                candidate = image_dir / f"{stem}{suf}"
                if candidate.exists():
                    side = candidate
                    break
        
        meta: Dict[str, Optional[str]] = {}
        if side is not None:
            meta = extract_from_xml(side)
        
        year = extract_year_from_name(img_name)
        rows.append(
            {
                "image_path": str(p),
                "image_filename": img_name,
                "year": year,
                **meta,
            }
        )

    df = pd.DataFrame(rows)
    # Normalize numeric columns
    for col in ["year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def atomic_write_csv(df, out: Path) -> None:
    """Write CSV atomically with backup of existing file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file
    if out.exists():
        bak = out.with_suffix(f".backup_{int(time.time())}.csv")
        shutil.copy2(out, bak)
    
    # Write to temp and move
    fd, tmpname = tempfile.mkstemp(dir=str(out.parent), prefix=".new_all_tiles_")
    os.close(fd)
    df.to_csv(tmpname, index=False)
    os.replace(tmpname, out)


def write_provenance(src: Path, out: Path, rows: int) -> None:
    """Write provenance JSON for generated CSV."""
    prov = {
        "generated_at": int(time.time()),
        "source": str(src.name),
        "source_sha256": sha256_of_file(src) if src.exists() else None,
        "rows": rows,
    }
    (out.parent / (out.stem + "_provenance.json")).write_text(json.dumps(prov, indent=2))


def choose_source(candidates: list[Path]) -> Optional[Path]:
    """Choose first existing source from candidates."""
    for p in candidates:
        if p.exists():
            return p
    return None


def build_tiles(image_dir: str | Path, out: str | Path, force_source: Optional[str] = None) -> int:
    """Main build function.
    
    Args:
        image_dir: Directory containing image files
        out: Output CSV path
        force_source: Optional source CSV for provenance
        
    Returns:
        Exit code (0 = success)
    """
    image_dir = Path(image_dir)
    out = Path(out)
    
    # Build DataFrame from image scan
    df = build_dataframe(image_dir)
    
    # Determine source for provenance
    src = None
    if force_source:
        src = Path(force_source)
    else:
        candidates = [
            out.parent / "all_tiles.csv",
            out.parent / "tiles.csv",
        ]
        src = choose_source(candidates)
    
    # Write CSV atomically and provenance
    atomic_write_csv(df, out)
    if src:
        write_provenance(src, out, len(df))
    else:
        # Create minimal provenance indicating only image-dir
        prov = {
            "generated_at": int(time.time()),
            "source": "images_scanned",
            "image_dir": str(image_dir),
            "rows": len(df),
        }
        (out.parent / (out.stem + "_provenance.json")).write_text(json.dumps(prov, indent=2))

    print(f"Wrote {out} ({len(df)} rows)")
    return 0


@cli_command(
    "build-tiles",
    help="Build new_all_tiles.csv from image directory scan",
    args={
        "image_dir": {
            "type": str,
            "required": True,
            "help": "Directory containing image files",
        },
        "out": {
            "type": str,
            "default": "data/new_all_tiles.csv",
            "help": "Output CSV path",
        },
        "force_source": {
            "type": str,
            "default": None,
            "help": "Force a specific source CSV to use for provenance",
        },
    },
)
def main(image_dir: str, out: str = "data/new_all_tiles.csv", force_source: str | None = None) -> int:
    """CLI entry point."""
    
    return build_tiles(
        image_dir=image_dir,
        out=out,
        force_source=force_source
    )


if __name__ == "__main__":
    raise SystemExit(main())
