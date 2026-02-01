#!/usr/bin/env python3
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SIDECAR_SUFFIXES = [".aux.xml", ".xml"]


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_xml(xml_path: Path) -> Dict[str, Optional[str]]:
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return {}

    meta: Dict[str, Optional[str]] = {}

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
                "image_path": str(p),
                "image_filename": img_name,
                "year": year,
            }
        )

    df = pd.DataFrame(rows)
    # normalize numeric columns
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
    parser.add_argument("--force-source", help="Force a specific source CSV to use for provenance")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    image_dir = Path(args.image_dir)
    out = Path(args.out)

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
