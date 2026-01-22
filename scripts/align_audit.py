"""Alignment Audit (M2.5)

CLI tool that checks alignment between CSV tile metadata and raster sidecars (.aux.xml).

Behavior:
- Reads CSV (default: data/new_all_tiles.csv) and expects columns `left,top,right,bottom,image_path`.
- For each row, computes the CSV centroid (lon/lat, EPSG:4326).
- Looks for an aux.xml sidecar next to the raster (image_path + '.aux.xml') or in a provided aux-dir.
- Extracts raster bounds from the aux.xml (attempts rasterio if available, else XML parsing).
- Reprojects both centroids and bbox centers to target CRS (default: EPSG:25832) and computes offsets in meters.
- Produces a JSON summary report and saves a PNG map with CSV vs. raster centroids and outliers.

Usage example:
    python scripts/align_audit.py --csv data/new_all_tiles.csv --out outputs/align_audit.json

"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from pyproj import Transformer
except Exception as e:  # pragma: no cover - import errors should be visible
    raise ImportError(
        "This script requires geopandas, shapely and pyproj. Please install the geo stack (see requirements)."
    )

try:
    import rasterio
    from rasterio.coords import BoundingBox
except Exception:
    rasterio = None  # optional, we fallback to XML parsing

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alignment audit: CSV vs .aux.xml bounds")
    p.add_argument("--csv", default="data/new_all_tiles.csv", help="CSV with tile metadata")
    p.add_argument("--base-dir", default=".", help="Base dir for image paths in CSV")
    p.add_argument("--aux-dir", default=None, help="Optional directory to look for .aux.xml files")
    p.add_argument("--target-crs", default="EPSG:25832", help="Target CRS for metric comparisons")
    p.add_argument("--out", default=None, help="Path for JSON report (default: outputs/align_audit_<date>.json)")
    p.add_argument("--plot", default=None, help="Path for PNG plot (default: outputs/align_audit_<date>.png)")
    p.add_argument("--max-offset-m", type=float, default=1000.0, help="Threshold (m) for outlier reporting")
    return p.parse_args()


def find_aux_for_image(image_path: Path, aux_dir: Optional[Path] = None) -> Optional[Path]:
    # Common locations: image_path + '.aux.xml', same dir, aux_dir with same basename + '.aux.xml'
    candidates = []
    candidates.append(image_path.with_suffix(image_path.suffix + ".aux.xml"))
    candidates.append(image_path.with_suffix(".aux.xml"))
    if aux_dir:
        candidates.append(aux_dir / (image_path.stem + ".aux.xml"))
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_aux_bbox_xml(xml_path: Path) -> Optional[Tuple[float, float, float, float, str]]:
    """Attempt to parse bounding box from .aux.xml. Returns (minx, miny, maxx, maxy, crs_wkt_or_epsg)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Try GeoTransform first
        gt = None
        for elem in root.iter():
            if elem.tag.lower().endswith("geotransform"):
                txt = (elem.text or "").strip()
                nums = [float(x) for x in txt.split()
                        if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if len(nums) >= 6:
                    gt = nums[:6]
                    break
        # Try corner coordinates
        ul = lr = None
        for elem in root.iter():
            tag = elem.tag.lower()
            if tag.endswith("upperleftx"):
                ulx = float(elem.text.strip())
                # find sibling upperlefty
                uly = float(elem.getparent().find('UpperLeftY').text) if hasattr(elem, 'getparent') else None
            if tag.endswith("upperleft"):
                # e.g. <UpperLeft> 5.8333 51.875 </UpperLeft>
                parts = (elem.text or "").split()
                if len(parts) >= 2:
                    ul = (float(parts[0]), float(parts[1]))
            if tag.endswith("lowerright"):
                parts = (elem.text or "").split()
                if len(parts) >= 2:
                    lr = (float(parts[0]), float(parts[1]))
        # If GeoTransform present and we can get size from image via rasterio or from XML, try to compute bbox
        if gt is not None and rasterio is not None:
            # Not enough info here to compute corners without raster size; fallback to searching for explicit corner coords
            pass
        if ul and lr:
            minx = min(ul[0], lr[0])
            maxx = max(ul[0], lr[0])
            miny = min(ul[1], lr[1])
            maxy = max(ul[1], lr[1])
            # Attempt to find SRS
            srs = None
            for elem in root.iter():
                if elem.tag.lower().endswith('srs') or elem.tag.lower().endswith('spatialreference'):
                    srs = (elem.text or '').strip()
                    break
            return (minx, miny, maxx, maxy, srs)
    except ET.ParseError:
        return None
    return None


def aux_bbox_via_rasterio(image_path: Path) -> Optional[Tuple[float, float, float, float, str]]:
    if rasterio is None:
        return None
    try:
        # try to open sidecar via rasterio (some GANs allow opening png with world file but not always)
        with rasterio.open(str(image_path)) as ds:
            b = ds.bounds
            crs = ds.crs.to_string() if ds.crs is not None else None
            return (b.left, b.bottom, b.right, b.top, crs)
    except Exception:
        return None


def make_report(csv_path: Path, base_dir: Path, aux_dir: Optional[Path], target_crs: str, max_offset_m: float, out_json: Path, out_png: Optional[Path]):
    df = pd.read_csv(csv_path)
    required_cols = {"left", "right", "top", "bottom", "image_path"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    # compute centroid lon/lat
    df = df.copy()
    df["centroid_lon"] = (df["left"] + df["right"]) / 2.0
    df["centroid_lat"] = (df["top"] + df["bottom"]) / 2.0

    gdf_csv = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.centroid_lon, df.centroid_lat)], crs="EPSG:4326")

    results = []

    # transformer to target CRS
    transformer_to_target = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    for idx, row in gdf_csv.iterrows():
        image_path = (base_dir / row["image_path"]).resolve()
        aux_path = find_aux_for_image(image_path, aux_dir=aux_dir)
        csv_centroid_proj = transformer_to_target.transform(row["centroid_lon"], row["centroid_lat"]) if transformer_to_target else None
        rec: Dict = {
            "index": int(idx),
            "image_path": str(image_path),
            "csv_centroid_lonlat": [row["centroid_lon"], row["centroid_lat"]],
            "csv_centroid_proj": [csv_centroid_proj[0], csv_centroid_proj[1]] if csv_centroid_proj else None,
            "aux_found": False,
            "aux_bbox": None,
            "aux_centroid_proj": None,
            "offset_m": None,
        }
        if aux_path:
            rec["aux_found"] = True
            # try rasterio first
            bbox = aux_bbox_via_rasterio(aux_path) if aux_path is not None else None
            if not bbox:
                # Attempt to parse aux.xml directly
                parsed = parse_aux_bbox_xml(aux_path)
                if parsed:
                    bbox = (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4])
            if bbox:
                minx, miny, maxx, maxy, srs = bbox
                rec["aux_bbox"] = [minx, miny, maxx, maxy]
                aux_cx = (minx + maxx) / 2.0
                aux_cy = (miny + maxy) / 2.0
                # Need to detect aux CRS: if srs contains '3857' or indicates Mercator, assume EPSG:3857
                aux_crs = None
                if srs and isinstance(srs, str) and ("3857" in srs or "Mercator" in srs):
                    aux_crs = "EPSG:3857"
                else:
                    # Attempt heuristic: if values are in 1e6..2e7 range assume EPSG:3857
                    if abs(aux_cx) > 1e5 or abs(aux_cy) > 1e5:
                        aux_crs = "EPSG:3857"
                    else:
                        aux_crs = "EPSG:4326"
                # transform aux centroid to target
                transformer_aux_to_target = Transformer.from_crs(aux_crs, target_crs, always_xy=True)
                aux_cx_t, aux_cy_t = transformer_aux_to_target.transform(aux_cx, aux_cy)
                rec["aux_centroid_proj"] = [aux_cx_t, aux_cy_t]
                # compute Euclidean distance
                dx = csv_centroid_proj[0] - aux_cx_t
                dy = csv_centroid_proj[1] - aux_cy_t
                offset = math.hypot(dx, dy)
                rec["offset_m"] = offset
        results.append(rec)

    # metrics
    offsets = [r["offset_m"] for r in results if r["offset_m"] is not None]
    n_total = len(results)
    n_aux = sum(1 for r in results if r["aux_found"])
    n_with_offset = sum(1 for r in results if r["offset_m"] is not None)
    outliers = [r for r in results if r["offset_m"] is not None and r["offset_m"] > max_offset_m]

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "csv": str(csv_path),
        "n_total": n_total,
        "n_aux_found": n_aux,
        "n_offsets": n_with_offset,
        "offset_m_median": float(np.median(offsets)) if offsets else None,
        "offset_m_max": float(np.max(offsets)) if offsets else None,
        "n_outliers": len(outliers),
        "outliers": [{"index": r["index"], "image_path": r["image_path"], "offset_m": r["offset_m"]} for r in outliers[:100]],
    }

    report = {"summary": summary, "results": results}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)

    if out_png:
        # Plot a simple scatter in target CRS for the rows that have both points
        pts_csv = []
        pts_aux = []
        labels = []
        for r in results:
            if r["csv_centroid_proj"] and r["aux_centroid_proj"]:
                pts_csv.append(r["csv_centroid_proj"])
                pts_aux.append(r["aux_centroid_proj"])
                labels.append(os.path.basename(r["image_path"]))
        if pts_csv:
            xs_csv, ys_csv = zip(*pts_csv)
            xs_aux, ys_aux = zip(*pts_aux)
            plt.figure(figsize=(10, 10))
            plt.scatter(xs_csv, ys_csv, c="tab:blue", s=10, label="CSV centroids")
            plt.scatter(xs_aux, ys_aux, c="tab:orange", s=10, label="aux centroids")
            for i, (x1, y1) in enumerate(pts_csv):
                x2, y2 = pts_aux[i]
                plt.plot([x1, x2], [y1, y2], c="0.7", linewidth=0.5)
            plt.legend()
            plt.title("CSV vs aux centroids ({}), outliers>{} m".format(target_crs, max_offset_m))
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis("equal")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()

    return report


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    base_dir = Path(args.base_dir)
    aux_dir = Path(args.aux_dir) if args.aux_dir else None
    target_crs = args.target_crs
    out_json = Path(args.out) if args.out else Path("outputs") / f"align_audit_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    out_png = Path(args.plot) if args.plot else Path(str(out_json).replace('.json', '.png'))

    report = make_report(csv_path, base_dir, aux_dir, target_crs, args.max_offset_m, out_json, out_png)
    print(f"Wrote report: {out_json}")


if __name__ == "__main__":
    main()
