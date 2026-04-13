from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from .models import (
    DEBUG_MASK_MANIFEST_FILENAME,
    SUPPORTED_CLASSES,
    normalize_class,
    repo_root,
    require_columns,
    resolve_path,
    utc_now,
    write_json,
)
from .prepare import (
    build_patch_contexts,
    extract_lines,
    load_roads_layer,
    load_selected_patches,
    roads_for_crs,
    subset_by_bounds,
)
from .runs import resolve_roads_layer_name


def load_patch_mask_requirements(handoff_dir: Path) -> dict[str, str]:
    requirements_path = handoff_dir / "patch_mask_requirements.csv"
    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Missing patch_mask_requirements.csv: {requirements_path}"
        )
    df = pd.read_csv(requirements_path)
    require_columns(
        df,
        ["patch_id", "required_mask_filename"],
        "patch_mask_requirements.csv",
    )
    requirements: dict[str, str] = {}
    for row in df.to_dict("records"):
        patch_id = str(row.get("patch_id", "")).strip()
        filename = str(row.get("required_mask_filename", "")).strip()
        if patch_id:
            requirements[patch_id] = filename
    return requirements


def render_mask_for_patch(
    patch: Any,
    roads_gdf: Any,
    *,
    class_widths_px: Mapping[int, int],
) -> np.ndarray:
    subset = subset_by_bounds(roads_gdf, patch.bounds)
    patch_poly = box(*patch.bounds)
    shapes: list[tuple[Any, int]] = []
    for row in subset.to_dict("records"):
        class_id = normalize_class(row.get("class"))
        if class_id not in class_widths_px:
            continue
        width_px = int(class_widths_px[class_id])
        if width_px <= 0:
            continue
        clipped = row["geometry"].intersection(patch_poly)
        if clipped.is_empty:
            continue
        buffer_distance = (float(width_px) * float(patch.pixel_size)) / 2.0
        for line in extract_lines(clipped):
            if line.length <= 0:
                continue
            polygon = line.buffer(buffer_distance)
            if polygon.is_empty:
                continue
            shapes.append((polygon.intersection(patch_poly), 255))
    if not shapes:
        return np.zeros((patch.height, patch.width), dtype=np.uint8)
    return rasterize(
        shapes=shapes,
        out_shape=(patch.height, patch.width),
        transform=patch.transform,
        fill=0,
        dtype="uint8",
    )


def write_binary_mask_geotiff(mask: np.ndarray, *, patch: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=int(patch.height),
        width=int(patch.width),
        count=1,
        dtype="uint8",
        transform=patch.transform,
        crs=patch.crs_wkt or None,
        compress="deflate",
    ) as dst:
        dst.write(np.asarray(mask, dtype=np.uint8), 1)


def render_width_calibration_debug_masks(
    *,
    handoff_dir: str | Path,
    roads_gpkg: str | Path,
    out_dir: str | Path,
    fixed_width_px: int,
    roads_layer: str | None = None,
) -> dict[str, Any]:
    fixed_width_value = int(fixed_width_px)
    if fixed_width_value <= 0:
        raise ValueError("fixed_width_px must be a positive integer.")
    repo_root_path = repo_root()
    handoff_dir_path = resolve_path(
        handoff_dir, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    roads_gpkg_path = resolve_path(
        roads_gpkg, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    out_dir_path = resolve_path(
        out_dir, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    resolved_roads_layer = (
        str(roads_layer).strip()
        if roads_layer is not None and str(roads_layer).strip()
        else resolve_roads_layer_name(roads_gpkg_path)
    )
    selected_df = load_selected_patches(handoff_dir_path, exclude_hamburg=False)
    patch_contexts = build_patch_contexts(handoff_dir_path, selected_df)
    mask_requirements = load_patch_mask_requirements(handoff_dir_path)
    missing_patch_ids = [
        patch.patch_id
        for patch in patch_contexts
        if patch.patch_id not in mask_requirements
    ]
    if missing_patch_ids:
        raise ValueError(
            "patch_mask_requirements.csv is missing required_mask_filename entries for patch_id(s): "
            f"{missing_patch_ids}"
        )
    empty_filename_patch_ids = [
        patch.patch_id
        for patch in patch_contexts
        if not str(mask_requirements.get(patch.patch_id, "")).strip()
    ]
    if empty_filename_patch_ids:
        raise ValueError(
            "patch_mask_requirements.csv has empty required_mask_filename entries for patch_id(s): "
            f"{empty_filename_patch_ids}"
        )
    roads_gdf = load_roads_layer(roads_gpkg_path, roads_layer=resolved_roads_layer)
    class_widths = {class_id: fixed_width_value for class_id in SUPPORTED_CLASSES}
    cache: dict[str, Any] = {}
    written_masks: list[str] = []
    for patch in patch_contexts:
        patch_roads = roads_for_crs(roads_gdf, patch.crs_wkt, cache)
        mask = render_mask_for_patch(
            patch,
            patch_roads,
            class_widths_px=class_widths,
        )
        out_path = out_dir_path / str(mask_requirements[patch.patch_id])
        write_binary_mask_geotiff(mask, patch=patch, out_path=out_path)
        written_masks.append(str(out_path))
    manifest_path = out_dir_path / DEBUG_MASK_MANIFEST_FILENAME
    write_json(
        manifest_path,
        {
            "debug_only": True,
            "test_only": True,
            "rendering_mode": "fixed_width_px",
            "fixed_width_px": fixed_width_value,
            "handoff_dir": str(handoff_dir_path),
            "roads_gpkg": str(roads_gpkg_path),
            "roads_layer": resolved_roads_layer,
            "generated_utc": utc_now(),
            "mask_count": len(written_masks),
        },
    )
    return {
        "out_dir": str(out_dir_path),
        "manifest_json": str(manifest_path),
        "mask_count": len(written_masks),
        "fixed_width_px": fixed_width_value,
    }


__all__ = [
    "load_patch_mask_requirements",
    "render_mask_for_patch",
    "render_width_calibration_debug_masks",
    "write_binary_mask_geotiff",
]
