"""Spatial alignment audit tool."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command
from dataselector.data.spatial_schema import (
    coordinates_look_projected,
    normalize_spatial_schema,
)

try:
    import matplotlib.pyplot as plt
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    raise ImportError(
        "This tool requires geopandas, shapely, pyproj and matplotlib. "
        "Please install the geo stack (see requirements-geo.txt)."
    )

try:
    import rasterio
except ImportError:
    rasterio = None  # optional, fallback to XML parsing

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def find_aux_xml(image_path: Path, aux_dir: Optional[Path] = None) -> Optional[Path]:
    """Find .aux.xml sidecar for image file.

    Args:
        image_path: Path to image file
        aux_dir: Optional directory to search for .aux.xml files

    Returns:
        Path to .aux.xml file if found, None otherwise
    """
    candidates = [
        image_path.with_suffix(image_path.suffix + ".aux.xml"),
        image_path.with_suffix(".aux.xml"),
    ]

    if aux_dir:
        candidates.extend(
            [
                aux_dir / (image_path.name + ".aux.xml"),
                aux_dir / (image_path.stem + ".aux.xml"),
                aux_dir / (image_path.stem + image_path.suffix + ".aux.xml"),
            ]
        )

    for c in candidates:
        if c.exists():
            return c

    return None


def extract_bounds_from_aux(
    aux_path: Path,
) -> Optional[Tuple[float, float, float, float]]:
    """Extract raster bounds from .aux.xml file.

    Args:
        aux_path: Path to .aux.xml file

    Returns:
        Tuple of (left, top, right, bottom) in EPSG:4326, or None
    """
    try:
        tree = ET.parse(aux_path)
        root = tree.getroot()
        # Try to find GeoTransform
        geo_elem = root.find(".//GeoTransform")
        if geo_elem is not None and geo_elem.text:
            # Format: x_origin, pixel_width, 0, y_origin, 0, pixel_height
            values = [float(v.strip()) for v in geo_elem.text.split(",")]
            if len(values) == 6:
                x_origin, pixel_width, _, y_origin, _, pixel_height = values
                # Get raster size
                size_elem = root.find(".//RasterXSize")
                size_y_elem = root.find(".//RasterYSize")
                if size_elem is not None and size_y_elem is not None:
                    width = int(size_elem.text)
                    height = int(size_y_elem.text)
                    left = x_origin
                    top = y_origin
                    right = x_origin + (width * pixel_width)
                    bottom = y_origin + (height * pixel_height)
                    return (left, top, right, bottom)
    except (ET.ParseError, ValueError, TypeError, OSError) as exc:
        logger.warning("Failed to parse aux.xml '%s': %s", aux_path, exc)
    return None


@cli_command(
    "align-audit",
    help="Audit CSV vs raster alignment",
    args={
        "csv": {
            "type": str,
            "default": "data/new_all_tiles.csv",
            "help": "CSV with tile metadata",
        },
        "base_dir": {
            "type": str,
            "default": ".",
            "help": "Base dir for image paths",
        },
        "aux_dir": {
            "type": str,
            "default": None,
            "help": "Optional directory for .aux.xml files",
        },
        "target_crs": {
            "type": str,
            "default": "EPSG:25832",
            "help": "Target CRS for metric comparisons",
        },
        "max_offset_m": {
            "type": float,
            "default": 1000.0,
            "help": "Threshold (m) for outlier reporting",
        },
        "out": {
            "type": str,
            "default": None,
            "help": "Path for JSON report",
        },
        "plot": {
            "type": str,
            "default": None,
            "help": "Path for PNG plot",
        },
    },
)
def align_audit(
    csv: str = "data/new_all_tiles.csv",
    base_dir: str = ".",
    aux_dir: Optional[str] = None,
    target_crs: str = "EPSG:25832",
    max_offset_m: float = 1000.0,
    out: Optional[str] = None,
    plot: Optional[str] = None,
) -> int:
    """Audit alignment between CSV metadata and raster aux.xml sidecars.

    Args:
        csv: Path to CSV with tile metadata
        base_dir: Base directory for image paths
        aux_dir: Optional directory for .aux.xml files
        target_crs: Target CRS for metric comparisons (default: EPSG:25832)
        max_offset_m: Threshold in meters for outlier reporting
        out: Output path for JSON report
        plot: Output path for PNG plot

    Returns:
        0 on success, 1 on errors
    """
    csv_path = csv
    out_json = out
    out_plot = plot
    csv_path = Path(csv_path)
    base_dir = Path(base_dir)
    aux_dir = Path(aux_dir) if aux_dir else None

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    df = normalize_spatial_schema(df, require_bounds=True, copy=False)

    required_cols = ["ul_x", "ul_y", "lr_x", "lr_y", "image_path"]
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV missing required columns: {required_cols}")
        return 1

    csv_is_projected = coordinates_look_projected(df)
    transformer = None
    if not csv_is_projected:
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    results = []
    outliers = []

    for idx, row in df.iterrows():
        image_path = base_dir / row["image_path"]
        csv_center_x = row["center_x"]
        csv_center_y = row["center_y"]

        # Find aux.xml
        aux_path = find_aux_xml(image_path, aux_dir)
        if not aux_path:
            results.append(
                {
                    "index": int(idx),
                    "image_path": str(image_path),
                    "status": "missing_aux",
                    "offset_m": None,
                }
            )
            continue

        # Extract bounds
        bounds = extract_bounds_from_aux(aux_path)
        if not bounds:
            results.append(
                {
                    "index": int(idx),
                    "image_path": str(image_path),
                    "status": "parse_error",
                    "offset_m": None,
                }
            )
            continue

        aux_left, aux_top, aux_right, aux_bottom = bounds
        aux_center_x = (aux_left + aux_right) / 2
        aux_center_y = (aux_top + aux_bottom) / 2

        if csv_is_projected:
            csv_x, csv_y = csv_center_x, csv_center_y
            aux_x, aux_y = aux_center_x, aux_center_y
        else:
            # Transform both centroids to target CRS
            csv_x, csv_y = transformer.transform(csv_center_x, csv_center_y)
            aux_x, aux_y = transformer.transform(aux_center_x, aux_center_y)

        # Compute offset in meters
        offset_m = np.sqrt((csv_x - aux_x) ** 2 + (csv_y - aux_y) ** 2)

        result = {
            "index": int(idx),
            "image_path": str(image_path),
            "status": "ok",
            "offset_m": float(offset_m),
            "csv_center": [csv_center_x, csv_center_y],
            "aux_center": [aux_center_x, aux_center_y],
        }
        results.append(result)

        if offset_m > max_offset_m:
            outliers.append(result)

    # Create summary report
    report = {
        "timestamp": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "target_crs": target_crs,
        "max_offset_m": max_offset_m,
        "total_tiles": len(df),
        "audited": len([r for r in results if r["status"] == "ok"]),
        "outliers": len(outliers),
        "missing_aux": len([r for r in results if r["status"] == "missing_aux"]),
        "parse_errors": len([r for r in results if r["status"] == "parse_error"]),
        "results": results,
    }

    # Save JSON report
    if not out_json:
        out_json = (
            f"outputs/align_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {out_json_path}")

    # Create plot
    if out_plot:
        ok_results = [r for r in results if r["status"] == "ok"]
        if ok_results:
            offsets = [r["offset_m"] for r in ok_results]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(offsets, bins=50, edgecolor="black")
            ax.axvline(
                max_offset_m,
                color="r",
                linestyle="--",
                label=f"Threshold: {max_offset_m}m",
            )
            ax.set_xlabel("Offset (m)")
            ax.set_ylabel("Count")
            ax.set_title(f"CSV vs AUX Centroid Offsets (n={len(ok_results)})")
            ax.legend()
            out_plot_path = Path(out_plot)
            out_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {out_plot_path}")

    # Print summary
    print("\nAlignment Audit Summary:")
    print(f"  Total tiles: {report['total_tiles']}")
    print(f"  Audited: {report['audited']}")
    print(f"  Outliers (>{max_offset_m}m): {report['outliers']}")
    print(f"  Missing aux.xml: {report['missing_aux']}")
    print(f"  Parse errors: {report['parse_errors']}")

    return 0
