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
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from dataselector.cli_decorators import cli_command

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SIDECAR_SUFFIXES = [".aux.xml", ".xml"]
CITY_FROM_LONGNAME_RE = re.compile(r"^KDR_\d+[A-Z]?[_-](.+)$", re.IGNORECASE)
CITY_YEAR_SUFFIX_RE = re.compile(r"(?i)[\s._-]*(?:ca[\s._-]*)?(?:17|18|19|20)\d{2}$")
CITY_NOYEAR_SUFFIX_RE = re.compile(r"(?i)[\s._-]*o\.?\s*j\.?$")
SHORTNAME_BASE_RE = re.compile(r"^(KDR_\d+)[A-Z]$", re.IGNORECASE)
DEFAULT_NAME_SOURCE = Path("data/KDR100_foliage_with_files_epsg3857.csv")
DEFAULT_CITY_OVERRIDES = Path("data/city_overrides.csv")


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


def parse_geotransform(
    value: str,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Parse GDAL GeoTransform string into six floats."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) < 6:
        return None
    try:
        return tuple(float(p) for p in parts[:6])  # type: ignore[return-value]
    except Exception:
        return None


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    """Read image width/height in pixels (best-effort)."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height)
    except Exception:
        return None


def extract_year_from_name(name: str) -> Optional[int]:
    """Extract 4-digit year from filename."""
    import re

    m = re.search(r"(17|18|19|20)\d{2}", name)
    if m:
        return int(m.group(0))
    return None


def normalize_shortname(value: str | None) -> str:
    """Normalize a shortName to a stable merge key."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    return s.upper()


def shortname_base(value: str | None) -> str:
    """Return normalized base key (e.g. KDR_155B -> KDR_155)."""
    norm = normalize_shortname(value)
    m = SHORTNAME_BASE_RE.match(norm)
    if m:
        return m.group(1).upper()
    return norm


def extract_city_from_longname(name: str) -> Optional[str]:
    """Extract city token from KDR longName patterns with tolerant suffix handling."""
    raw = str(name).strip()
    if not raw:
        return None

    # Keep basename only.
    raw = raw.split("/")[-1].split("\\")[-1]
    raw = re.sub(r"\.[A-Za-z0-9]+$", "", raw)

    m = CITY_FROM_LONGNAME_RE.match(raw)
    if not m:
        return None
    city_tail = m.group(1).strip()

    # Remove known temporal suffix conventions.
    city_tail = CITY_NOYEAR_SUFFIX_RE.sub("", city_tail).strip()
    city_tail = CITY_YEAR_SUFFIX_RE.sub("", city_tail).strip()
    city_tail = city_tail.strip("._- ")

    return city_tail or None


def _normalize_source_columns(df):
    """Normalize legacy header patterns like 'shortName,C,254' to 'shortName'."""
    rename = {}
    for col in df.columns:
        base = str(col).strip().strip('"').split(",")[0].strip()
        rename[col] = base or str(col)
    return df.rename(columns=rename)


def load_name_enrichment(source_csv: Path):
    """Load optional metadata enrichment (short_norm/longName/city/year/city_source)."""
    import pandas as pd

    if not source_csv.exists():
        return None

    try:
        src = pd.read_csv(source_csv)
    except Exception:
        return None

    if src.empty:
        return None

    src = _normalize_source_columns(src)

    short_col = next(
        (
            c
            for c in ["shortName", "shortname", "short_name", "short"]
            if c in src.columns
        ),
        None,
    )
    long_col = next(
        (
            c
            for c in ["longName", "longname", "long_name", "filename", "file", "name"]
            if c in src.columns
        ),
        None,
    )
    if short_col is None and long_col is None:
        return None

    out = pd.DataFrame()
    if short_col is not None:
        out["short_raw"] = src[short_col].astype(str).str.strip()
    else:
        out["short_raw"] = (
            src[long_col].astype(str).str.extract(r"^(KDR_\d+[A-Z]?)", expand=False)
        )
    out["short_norm"] = out["short_raw"].apply(normalize_shortname)
    out = out[out["short_norm"] != ""].copy()

    if long_col is not None:
        out["longName"] = src[long_col].astype(str).str.strip()
        out["longName"] = (
            out["longName"]
            .str.replace(r"^.*/", "", regex=True)
            .str.replace(r"^.*\\", "", regex=True)
        )

    city_col = next(
        (c for c in ["city", "City", "ort", "Ort"] if c in src.columns), None
    )
    out["city"] = ""
    out["city_source"] = ""
    if city_col is not None:
        city_series = (
            src[city_col].astype(str).str.strip().replace({"nan": "", "None": ""})
        )
        out["city"] = city_series
        out.loc[out["city"] != "", "city_source"] = "epsg_source"
    if "longName" in out.columns:
        inferred = out["longName"].apply(extract_city_from_longname).fillna("")
        mask = out["city"].astype(str).str.strip() == ""
        out.loc[mask, "city"] = inferred[mask]
        out.loc[(out["city"] != "") & (out["city_source"] == ""), "city_source"] = (
            "longname_parse"
        )

    year_col = next(
        (c for c in ["year", "Year", "jahr", "Jahr"] if c in src.columns), None
    )
    if year_col is not None:
        out["year"] = pd.to_numeric(src[year_col], errors="coerce")
    elif "longName" in out.columns:
        out["year"] = out["longName"].apply(extract_year_from_name)

    # Keep first non-empty city per key, then first row.
    out["city_len"] = out["city"].astype(str).str.len()
    out = out.sort_values(["short_norm", "city_len"], ascending=[True, False])
    out = out.drop_duplicates(subset=["short_norm"], keep="first")
    out = out.drop(columns=["city_len"])

    keep_cols = ["short_norm"]
    for col in ["longName", "city", "year", "city_source"]:
        if col in out.columns:
            keep_cols.append(col)
    return out[keep_cols]


def load_city_overrides(path: Path):
    """Load manual city overrides with normalized keys."""
    import pandas as pd

    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "shortName" not in df.columns or "city" not in df.columns:
        return None

    out = df.copy()
    out["short_norm"] = out["shortName"].astype(str).apply(normalize_shortname)
    out["city"] = out["city"].fillna("").astype(str).str.strip()
    out = out[(out["short_norm"] != "") & (out["city"] != "")]
    if "source" not in out.columns:
        out["source"] = "manual_override"
    if "note" not in out.columns:
        out["note"] = ""
    out = out.drop_duplicates(subset=["short_norm"], keep="first")
    return out[["short_norm", "city", "source", "note"]]


def choose_best_city_backup(
    data_dir: Path, expected_rows: int, keys: list[str]
) -> Optional[Path]:
    """Pick backup CSV with highest usable city coverage for current keys."""
    import pandas as pd

    candidates = sorted(data_dir.glob("new_all_tiles.backup_*.csv"))
    best_path = None
    best_score = (-1, "")

    key_df = pd.DataFrame({"short_norm": keys})
    for p in candidates:
        try:
            b = pd.read_csv(p)
        except Exception:
            continue
        if (
            len(b) != expected_rows
            or "shortName" not in b.columns
            or "city" not in b.columns
        ):
            continue
        bmap = pd.DataFrame(
            {
                "short_norm": b["shortName"].astype(str).apply(normalize_shortname),
                "city": b["city"].fillna("").astype(str).str.strip(),
            }
        )
        bmap = bmap.drop_duplicates(subset=["short_norm"], keep="first")
        merged = key_df.merge(bmap, on="short_norm", how="left")
        score = int((merged["city"].fillna("").astype(str).str.strip() != "").sum())
        # Deterministic tie-breaker: lexical filename (later backup usually bigger timestamp).
        marker = p.name
        if (score, marker) > best_score:
            best_score = (score, marker)
            best_path = p
    return best_path


def build_dataframe(
    image_dir: Path,
    enrichment_source: Path | None = None,
    city_overrides_path: Path | None = None,
    backup_dir: Path | None = None,
):
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
            candidate = (
                p.with_suffix(p.suffix + suf)
                if suf.startswith(".") and not str(p).endswith(suf)
                else p.with_suffix(suf)
            )
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
        geo_transform_raw = meta.pop("GeoTransform", None)
        # Keep CSV compact; raw CRS WKT is not required for downstream loaders.
        meta.pop("SRS", None)

        width_px = None
        height_px = None
        pixel_width = None
        pixel_height = None
        ul_x = None
        ul_y = None
        lr_x = None
        lr_y = None
        data_quality = "unknown"

        gt = parse_geotransform(geo_transform_raw) if geo_transform_raw else None
        img_size = read_image_size(p)
        if img_size is not None:
            width_px, height_px = img_size

        if gt is not None and width_px is not None and height_px is not None:
            ul_x = gt[0]
            pixel_width = gt[1]
            ul_y = gt[3]
            pixel_height = gt[5]
            lr_x = ul_x + pixel_width * width_px
            lr_y = ul_y + pixel_height * height_px
            data_quality = "ok"
        elif gt is not None:
            ul_x = gt[0]
            pixel_width = gt[1]
            ul_y = gt[3]
            pixel_height = gt[5]
            data_quality = "partial"

        year = extract_year_from_name(img_name)
        rows.append(
            {
                "image_path": str(p),
                "image_filename": img_name,
                "filename": img_name,
                "year": year,
                "ul_x": ul_x,
                "ul_y": ul_y,
                "lr_x": lr_x,
                "lr_y": lr_y,
                "width_px": width_px,
                "height_px": height_px,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
                "data_quality": data_quality,
                **meta,
            }
        )

    df = pd.DataFrame(rows)

    # Keep a stable schema even when no images were discovered.
    # CI smoke tests rely on build-tiles succeeding for empty directories.
    required_columns = {
        "image_path": "",
        "image_filename": "",
        "filename": "",
        "year": None,
        "ul_x": None,
        "ul_y": None,
        "lr_x": None,
        "lr_y": None,
        "width_px": None,
        "height_px": None,
        "pixel_width": None,
        "pixel_height": None,
        "data_quality": "unknown",
    }
    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default

    if "shortName" not in df.columns:
        df["shortName"] = df["image_filename"].apply(
            lambda s: Path(str(s)).stem if pd.notna(s) else ""
        )
    else:
        df["shortName"] = df["shortName"].fillna("").astype(str).str.strip()
    df["short_norm"] = df["shortName"].apply(normalize_shortname)

    if "city" not in df.columns:
        df["city"] = ""
    else:
        df["city"] = df["city"].fillna("").astype(str).str.strip()
    if "city_source" not in df.columns:
        df["city_source"] = ""
    else:
        df["city_source"] = df["city_source"].fillna("").astype(str).str.strip()
    df.loc[(df["city"] != "") & (df["city_source"] == ""), "city_source"] = "sidecar"

    # Optional enrichment from legacy/canonical metadata sources.
    if enrichment_source is not None:
        enrich = load_name_enrichment(enrichment_source)
        if enrich is not None and not enrich.empty:
            # Exact key match first (case/suffix-normalized).
            df = df.merge(enrich, on="short_norm", how="left", suffixes=("", "_src"))

            if "longName_src" in df.columns:
                if "longName" not in df.columns:
                    df["longName"] = df["longName_src"]
                else:
                    base = df["longName"].fillna("").astype(str).str.strip()
                    src = df["longName_src"].fillna("").astype(str).str.strip()
                    df["longName"] = base.where(base != "", src)
                df.drop(columns=["longName_src"], inplace=True)

            if "city_src" in df.columns:
                if "city" not in df.columns:
                    df["city"] = df["city_src"]
                else:
                    base = df["city"].fillna("").astype(str).str.strip()
                    src = df["city_src"].fillna("").astype(str).str.strip()
                    df["city"] = base.where(base != "", src)
                if "city_source_src" in df.columns:
                    src_source = (
                        df["city_source_src"].fillna("").astype(str).str.strip()
                    )
                    base_source = df["city_source"].fillna("").astype(str).str.strip()
                    fill_mask = (base_source == "") & (
                        df["city"].fillna("").astype(str).str.strip() != ""
                    )
                    df.loc[fill_mask, "city_source"] = src_source[fill_mask]
                    df.loc[
                        fill_mask & (df["city_source"].fillna("") == ""), "city_source"
                    ] = "epsg_source"
                df.drop(columns=["city_src"], inplace=True)

            if "year_src" in df.columns:
                base = pd.to_numeric(df["year"], errors="coerce")
                src = pd.to_numeric(df["year_src"], errors="coerce")
                df["year"] = base.where(base.notna(), src)
                df.drop(columns=["year_src"], inplace=True)

            drop_src = [c for c in ["city_source_src"] if c in df.columns]
            if drop_src:
                df.drop(columns=drop_src, inplace=True)

            # Variant-base fallback (e.g. KDR_155B -> KDR_155) when exact mapping is absent.
            enrich_variant = enrich.copy()
            enrich_variant["short_base"] = enrich_variant["short_norm"].apply(
                shortname_base
            )
            enrich_variant["rank_exact"] = (
                enrich_variant["short_norm"] == enrich_variant["short_base"]
            ).astype(int)
            enrich_variant["rank_city"] = (
                enrich_variant["city"].fillna("").astype(str).str.strip() != ""
            ).astype(int)
            enrich_variant = enrich_variant.sort_values(
                ["short_base", "rank_exact", "rank_city"],
                ascending=[True, False, False],
            )
            enrich_variant = enrich_variant.drop_duplicates(
                subset=["short_base"], keep="first"
            )
            variant_cols = ["short_base", "longName", "city", "year", "city_source"]
            enrich_variant = enrich_variant[
                [c for c in variant_cols if c in enrich_variant.columns]
            ]
            enrich_variant = enrich_variant.rename(
                columns={
                    "longName": "longName_base",
                    "city": "city_base",
                    "year": "year_base",
                    "city_source": "city_source_base",
                }
            )
            df["short_base"] = df["short_norm"].apply(shortname_base)
            df = df.merge(enrich_variant, on="short_base", how="left")

            if "longName_base" in df.columns:
                base = df["longName"].fillna("").astype(str).str.strip()
                src = df["longName_base"].fillna("").astype(str).str.strip()
                df["longName"] = base.where(base != "", src)
            if "city_base" in df.columns:
                city_base = df["city"].fillna("").astype(str).str.strip()
                city_src = df["city_base"].fillna("").astype(str).str.strip()
                fill_mask = (city_base == "") & (city_src != "")
                df.loc[fill_mask, "city"] = city_src[fill_mask]
                df.loc[
                    fill_mask,
                    "city_source",
                ] = "variant_base"
            if "year_base" in df.columns:
                year_base = pd.to_numeric(df["year"], errors="coerce")
                year_src = pd.to_numeric(df["year_base"], errors="coerce")
                df["year"] = year_base.where(year_base.notna(), year_src)

            drop_variant = [
                c
                for c in [
                    "short_base",
                    "longName_base",
                    "city_base",
                    "year_base",
                    "city_source_base",
                ]
                if c in df.columns
            ]
            if drop_variant:
                df.drop(columns=drop_variant, inplace=True)

            # Fallback: derive city from longName if still missing
            if "longName" in df.columns:
                inferred_city = (
                    df["longName"]
                    .fillna("")
                    .astype(str)
                    .apply(extract_city_from_longname)
                )
                if "city" not in df.columns:
                    df["city"] = inferred_city
                else:
                    base = df["city"].fillna("").astype(str).str.strip()
                    fill = inferred_city.fillna("").astype(str).str.strip()
                    df["city"] = base.where(base != "", fill)
                    mask = (base == "") & (fill != "")
                    df.loc[mask, "city_source"] = "longname_parse"

                inferred_year = pd.to_numeric(
                    df["longName"].fillna("").astype(str).apply(extract_year_from_name),
                    errors="coerce",
                )
                if "year" not in df.columns:
                    df["year"] = inferred_year
                else:
                    base_year = pd.to_numeric(df["year"], errors="coerce")
                    df["year"] = base_year.where(base_year.notna(), inferred_year)

            # Keep common filename alias used by canonical metadata.
            if "filename" not in df.columns:
                df["filename"] = df["image_filename"]

    # Backup CSV fallback for unresolved city keys.
    missing_city_mask = df["city"].fillna("").astype(str).str.strip() == ""
    if missing_city_mask.any():
        candidate_dir = backup_dir if backup_dir is not None else image_dir.parent
        backup_path = choose_best_city_backup(
            candidate_dir, expected_rows=len(df), keys=df["short_norm"].tolist()
        )
        if backup_path is not None:
            backup_df = pd.read_csv(backup_path)
            backup_map = pd.DataFrame(
                {
                    "short_norm": backup_df["shortName"]
                    .astype(str)
                    .apply(normalize_shortname),
                    "city_backup": backup_df["city"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace({"nan": "", "None": ""}),
                }
            )
            backup_map = backup_map.drop_duplicates(subset=["short_norm"], keep="first")
            df = df.merge(backup_map, on="short_norm", how="left")
            base = df["city"].fillna("").astype(str).str.strip()
            src = df["city_backup"].fillna("").astype(str).str.strip()
            fill_mask = (base == "") & (src != "")
            df.loc[fill_mask, "city"] = src[fill_mask]
            df.loc[fill_mask, "city_source"] = "backup_fill"
            if "city_backup" in df.columns:
                df.drop(columns=["city_backup"], inplace=True)

    # Manual override for truly unresolved IDs.
    if city_overrides_path is not None:
        overrides = load_city_overrides(city_overrides_path)
        if overrides is not None and not overrides.empty:
            ov = overrides.rename(
                columns={
                    "city": "city_override",
                    "source": "city_override_source",
                    "note": "city_override_note",
                }
            )
            df = df.merge(ov, on="short_norm", how="left")
            override_city = df["city_override"].fillna("").astype(str).str.strip()
            apply_mask = override_city != ""
            df.loc[apply_mask, "city"] = override_city[apply_mask]
            df.loc[apply_mask, "city_source"] = "manual_override"
            drop_override = [
                c
                for c in ["city_override", "city_override_source", "city_override_note"]
                if c in df.columns
            ]
            if drop_override:
                df.drop(columns=drop_override, inplace=True)

    # Final normalization.
    if "city" in df.columns:
        df["city"] = (
            df["city"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": ""})
        )
    if "city_source" in df.columns:
        df["city_source"] = (
            df["city_source"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": ""})
        )

    # Normalize numeric columns
    for col in ["year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Internal merge key must not leak to canonical output.
    if "short_norm" in df.columns:
        df.drop(columns=["short_norm"], inplace=True)

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
    (out.parent / (out.stem + "_provenance.json")).write_text(
        json.dumps(prov, indent=2)
    )


def choose_source(candidates: list[Path]) -> Optional[Path]:
    """Choose first existing source from candidates."""
    for p in candidates:
        if p.exists():
            return p
    return None


def build_tiles(
    image_dir: str | Path,
    out: str | Path,
    force_source: Optional[str] = None,
    name_source_csv: Optional[str] = None,
    city_overrides: Optional[str] = None,
) -> int:
    """Main build function.

    Args:
        image_dir: Directory containing image files
        out: Output CSV path
        force_source: Optional source CSV for provenance
        name_source_csv: Optional enrichment CSV to map longName/city/year
        city_overrides: Optional manual overrides CSV for unresolved city keys

    Returns:
        Exit code (0 = success)
    """
    image_dir = Path(image_dir)
    out = Path(out)

    canonical_out = (Path.cwd() / "data/new_all_tiles.csv").resolve()
    is_canonical_target = out.resolve() == canonical_out

    # Determine optional enrichment source.
    enrichment_candidates = []
    if name_source_csv:
        enrichment_candidates.append(Path(name_source_csv))
    if force_source and not name_source_csv:
        enrichment_candidates.append(Path(force_source))
    if is_canonical_target and not name_source_csv:
        enrichment_candidates.append(DEFAULT_NAME_SOURCE)
    enrichment_candidates.extend(
        [out.parent / "all_tiles.csv", out.parent / "tiles.csv"]
    )
    enrichment_source = choose_source(enrichment_candidates)

    if city_overrides:
        city_overrides_path = Path(city_overrides)
    elif is_canonical_target:
        city_overrides_path = DEFAULT_CITY_OVERRIDES
    else:
        city_overrides_path = None

    # Build DataFrame from image scan
    df = build_dataframe(
        image_dir,
        enrichment_source=enrichment_source,
        city_overrides_path=city_overrides_path,
        backup_dir=out.parent,
    )

    # Canonical contract: city, source-trace and metric schema must be complete.
    if is_canonical_target:
        city_missing = (
            int((df["city"].fillna("").astype(str).str.strip() == "").sum())
            if "city" in df.columns
            else len(df)
        )
        source_missing = (
            int((df["city_source"].fillna("").astype(str).str.strip() == "").sum())
            if "city_source" in df.columns
            else len(df)
        )
        bounds_cols = ["ul_x", "ul_y", "lr_x", "lr_y"]
        bounds_missing = 0
        for col in bounds_cols:
            if col not in df.columns:
                bounds_missing += len(df)
            else:
                bounds_missing += int(df[col].isna().sum())
        if city_missing > 0:
            raise SystemExit(
                "Canonical city contract violated: {} unresolved city rows in {}. "
                "Add/adjust source mapping or data/city_overrides.csv.".format(
                    city_missing, out
                )
            )
        if source_missing > 0:
            raise SystemExit(
                "Canonical city contract violated: {} rows without city_source trace in {}.".format(
                    source_missing, out
                )
            )
        if bounds_missing > 0:
            raise SystemExit(
                "Canonical spatial contract violated: {} missing ul/lr bound values in {}.".format(
                    bounds_missing, out
                )
            )

    # Determine source for provenance
    src = None
    if force_source:
        src = Path(force_source)
    else:
        src = enrichment_source

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
        (out.parent / (out.stem + "_provenance.json")).write_text(
            json.dumps(prov, indent=2)
        )

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
        "name_source_csv": {
            "type": str,
            "default": None,
            "help": "Optional enrichment CSV for longName/city/year (canonical default: data/KDR100_foliage_with_files_epsg3857.csv)",
        },
        "city_overrides": {
            "type": str,
            "default": None,
            "help": "Optional CSV with manual city overrides (columns: shortName,city,source,note)",
        },
    },
)
def main(
    image_dir: str,
    out: str = "data/new_all_tiles.csv",
    force_source: str | None = None,
    name_source_csv: str | None = None,
    city_overrides: str | None = None,
) -> int:
    """CLI entry point."""

    return build_tiles(
        image_dir=image_dir,
        out=out,
        force_source=force_source,
        name_source_csv=name_source_csv,
        city_overrides=city_overrides,
    )


if __name__ == "__main__":
    raise SystemExit(main())
