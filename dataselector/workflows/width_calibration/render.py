from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from dataselector.runtime.parameter_snapshot import compute_file_sha256

from .models import (
    DEBUG_MASK_MANIFEST_FILENAME,
    FINAL_MASK_MANIFEST_FILENAME,
    SUMMARY_COLUMNS,
    SUPPORTED_CLASSES,
    WORKFLOW_VERSION,
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

UPSTREAM_FINAL_WIDTH_CONTRACT_FILENAME = "phase5_final_width_contract.json"
UPSTREAM_FINAL_WIDTH_SCHEMA_VERSION = "phase5_final_width_authority_v1"


def _infer_training_repo_root(handoff_dir: Path) -> Path | None:
    # Expected layout: <train_repo>/data/handoff/<handoff_id>
    if handoff_dir.parent.name != "handoff":
        return None
    data_dir = handoff_dir.parent.parent
    if data_dir.name != "data":
        return None
    return data_dir.parent


def _selection_id_from_handoff_manifest(handoff_dir: Path) -> str:
    manifest_path = handoff_dir / "patch_handoff_manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("selection_id", "")).strip()


def _prefer_relative(path_value: Path, *, anchor_root: Path | None) -> str:
    if anchor_root is None:
        return str(path_value)
    try:
        return str(path_value.relative_to(anchor_root))
    except Exception:
        return str(path_value)


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


def load_final_widths(summary_csv_path: Path) -> dict[int, int]:
    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"Missing width calibration summary CSV: {summary_csv_path}"
        )
    summary_df = pd.read_csv(summary_csv_path)
    require_columns(summary_df, SUMMARY_COLUMNS, "width_calibration_summary.csv")
    widths: dict[int, int] = {}
    unsupported_classes: list[int] = []
    invalid_classes: list[str] = []
    for row in summary_df.to_dict("records"):
        class_id = normalize_class(row.get("class"))
        if class_id is None:
            invalid_classes.append(str(row.get("class", "")))
            continue
        if class_id not in SUPPORTED_CLASSES:
            unsupported_classes.append(int(class_id))
            continue
        final_width_value = row.get("final_width_px")
        if pd.isna(final_width_value):
            raise ValueError(f"Missing final_width_px for class {class_id}.")
        width_px = int(final_width_value)
        if width_px <= 0:
            raise ValueError(
                f"final_width_px must be positive for class {class_id}: {width_px}"
            )
        widths[int(class_id)] = width_px
    if invalid_classes:
        raise ValueError(
            "width_calibration_summary.csv contains invalid class values: "
            f"{invalid_classes}"
        )
    if unsupported_classes:
        raise ValueError(
            "width_calibration_summary.csv contains unsupported class values: "
            f"{sorted(set(unsupported_classes))}"
        )
    missing_classes = sorted(
        class_id for class_id in SUPPORTED_CLASSES if class_id not in widths
    )
    if missing_classes:
        raise ValueError(
            "width_calibration_summary.csv is missing final_width_px for supported class(es): "
            f"{missing_classes}"
        )
    return widths


def validate_expected_sha256(
    *,
    path: Path,
    label: str,
    actual_sha256: str,
    expected_sha256: str | None,
) -> None:
    if expected_sha256 is None or not str(expected_sha256).strip():
        return
    expected_value = str(expected_sha256).strip().lower()
    actual_value = str(actual_sha256).strip().lower()
    if actual_value != expected_value:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: "
            f"expected {expected_value}, got {actual_value}"
        )


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


def render_width_calibration_final_masks(
    *,
    handoff_dir: str | Path,
    roads_gpkg: str | Path,
    summary_csv: str | Path,
    out_dir: str | Path,
    roads_layer: str | None = None,
    expected_roads_gpkg_sha256: str | None = None,
    expected_summary_csv_sha256: str | None = None,
    width_tasks_csv: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = repo_root()
    handoff_dir_path = resolve_path(
        handoff_dir, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    roads_gpkg_path = resolve_path(
        roads_gpkg, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    summary_csv_path = resolve_path(
        summary_csv, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    out_dir_path = resolve_path(
        out_dir, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    roads_gpkg_sha256 = compute_file_sha256(roads_gpkg_path)
    summary_csv_sha256 = compute_file_sha256(summary_csv_path)
    width_tasks_csv_path: Path | None = None
    width_tasks_sha256 = ""
    if width_tasks_csv is not None and str(width_tasks_csv).strip():
        width_tasks_csv_path = resolve_path(
            width_tasks_csv,
            repo_root_path=repo_root_path,
            prefer_repo=True,
        ).resolve()
        width_tasks_sha256 = compute_file_sha256(width_tasks_csv_path)
    validate_expected_sha256(
        path=roads_gpkg_path,
        label="roads_gpkg",
        actual_sha256=roads_gpkg_sha256,
        expected_sha256=expected_roads_gpkg_sha256,
    )
    validate_expected_sha256(
        path=summary_csv_path,
        label="summary_csv",
        actual_sha256=summary_csv_sha256,
        expected_sha256=expected_summary_csv_sha256,
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)
    resolved_roads_layer = (
        str(roads_layer).strip()
        if roads_layer is not None and str(roads_layer).strip()
        else resolve_roads_layer_name(roads_gpkg_path)
    )
    class_widths = load_final_widths(summary_csv_path)
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
    manifest_path = out_dir_path / FINAL_MASK_MANIFEST_FILENAME
    write_json(
        manifest_path,
        {
            "workflow_version": WORKFLOW_VERSION,
            "debug_only": False,
            "test_only": False,
            "rendering_mode": "final_width_px",
            "summary_csv": str(summary_csv_path),
            "summary_csv_sha256": summary_csv_sha256,
            "handoff_dir": str(handoff_dir_path),
            "roads_gpkg": str(roads_gpkg_path),
            "roads_gpkg_sha256": roads_gpkg_sha256,
            "roads_layer": resolved_roads_layer,
            "generated_utc": utc_now(),
            "mask_count": len(written_masks),
            "class_widths_px": {
                str(k): int(v) for k, v in sorted(class_widths.items())
            },
        },
    )
    training_repo_root = _infer_training_repo_root(handoff_dir_path)
    masks_dir_value = _prefer_relative(out_dir_path, anchor_root=training_repo_root)
    contract_path = handoff_dir_path / UPSTREAM_FINAL_WIDTH_CONTRACT_FILENAME
    write_json(
        contract_path,
        {
            "schema_version": UPSTREAM_FINAL_WIDTH_SCHEMA_VERSION,
            "status": "ready_for_training",
            "selection_id": _selection_id_from_handoff_manifest(handoff_dir_path),
            "source_handoff_id": str(handoff_dir_path.name),
            "mask_manifest": FINAL_MASK_MANIFEST_FILENAME,
            "mask_manifest_sha256": compute_file_sha256(manifest_path),
            "patch_count": len(written_masks),
            "class_widths_px": {
                str(k): int(v) for k, v in sorted(class_widths.items())
            },
            "width_summary_sha256": summary_csv_sha256,
            "roads_gpkg_sha256": roads_gpkg_sha256,
            "width_tasks_sha256": width_tasks_sha256,
            "masks_dir": masks_dir_value,
        },
    )
    return {
        "out_dir": str(out_dir_path),
        "manifest_json": str(manifest_path),
        "upstream_contract_json": str(contract_path),
        "mask_count": len(written_masks),
        "class_widths_px": {
            str(k): int(v) for k, v in sorted(class_widths.items())
        },
        "summary_csv_sha256": summary_csv_sha256,
        "roads_gpkg_sha256": roads_gpkg_sha256,
        "width_tasks_sha256": width_tasks_sha256,
        "width_tasks_csv": "" if width_tasks_csv_path is None else str(width_tasks_csv_path),
    }


__all__ = [
    "load_final_widths",
    "load_patch_mask_requirements",
    "render_mask_for_patch",
    "render_width_calibration_debug_masks",
    "render_width_calibration_final_masks",
    "validate_expected_sha256",
    "write_binary_mask_geotiff",
]
