from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from .models import (
    AUDIT_RARE_CLASSES,
    COMMON_CLASS_QUOTAS,
    SENSITIVITY_COLUMNS,
    SENSITIVITY_FILENAME,
    SENSITIVITY_OVERLAY_DIRNAME,
    SensitivityRow,
    normalize_class,
    repo_root,
    resolve_path,
)
from .prepare import (
    build_patch_contexts,
    extract_lines,
    load_roads_layer,
    load_selected_patches,
    roads_for_crs,
    subset_by_bounds,
)
from .render import render_mask_for_patch
from .runs import resolve_roads_layer_name


def compute_patch_class_presence(
    *, handoff_dir: Path, roads_gdf: Any
) -> dict[str, set[int]]:
    selected_df = load_selected_patches(handoff_dir)
    contexts = build_patch_contexts(handoff_dir, selected_df)
    cache: dict[str, Any] = {}
    patch_classes: dict[str, set[int]] = {}
    for patch in contexts:
        patch_roads = roads_for_crs(roads_gdf, patch.crs_wkt, cache)
        subset = subset_by_bounds(patch_roads, patch.bounds)
        patch_poly = box(*patch.bounds)
        classes: set[int] = set()
        for row in subset.to_dict("records"):
            class_id = normalize_class(row.get("class"))
            if class_id is None:
                continue
            clipped = row["geometry"].intersection(patch_poly)
            if clipped.is_empty:
                continue
            if extract_lines(clipped):
                classes.add(class_id)
        patch_classes[patch.patch_id] = classes
    return patch_classes


def connected_components(mask: np.ndarray) -> int:
    binary = np.asarray(mask > 0, dtype=bool)
    if binary.size == 0:
        return 0
    visited = np.zeros(binary.shape, dtype=bool)
    count = 0
    height, width = binary.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in offsets:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


def load_patch_context_map(handoff_dir: Path) -> dict[str, Any]:
    return {
        patch.patch_id: patch
        for patch in build_patch_contexts(
            handoff_dir, load_selected_patches(handoff_dir)
        )
    }


def select_audit_subset(patch_classes: dict[str, set[int]]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for class_id in sorted(AUDIT_RARE_CLASSES):
        candidates = sorted(
            patch_id
            for patch_id, classes in patch_classes.items()
            if class_id in classes
        )
        if candidates and candidates[0] not in seen:
            selected.append((candidates[0], f"rare_class_{class_id}"))
            seen.add(candidates[0])
    common_candidates = sorted(
        patch_id
        for patch_id, classes in patch_classes.items()
        if any(class_id in COMMON_CLASS_QUOTAS for class_id in classes)
    )
    for patch_id in common_candidates:
        if patch_id in seen:
            continue
        selected.append((patch_id, "common_pool"))
        seen.add(patch_id)
        if sum(1 for _patch_id, reason in selected if reason == "common_pool") >= 2:
            break
    return selected


def save_sensitivity_overlay(
    *,
    patch: Any,
    handoff_dir: Path,
    overlay_dir: Path,
    baseline_mask: np.ndarray,
    minus_mask: np.ndarray,
    plus_mask: np.ndarray,
    counts: dict[str, dict[str, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quicklook_path = (handoff_dir / patch.quicklook_relpath).resolve()
    with rasterio.open(quicklook_path) as ds:
        image = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("Quicklook", image, None, None),
        (
            f"Baseline\nfg={counts['baseline']['foreground_pixels']} cc={counts['baseline']['connected_components']}",
            image,
            baseline_mask,
            "Reds",
        ),
        (
            f"Median-1px\nfg={counts['median_minus_1px']['foreground_pixels']} cc={counts['median_minus_1px']['connected_components']}",
            image,
            minus_mask,
            "Blues",
        ),
        (
            f"Median+1px\nfg={counts['median_plus_1px']['foreground_pixels']} cc={counts['median_plus_1px']['connected_components']}",
            image,
            plus_mask,
            "Greens",
        ),
    ]
    for ax, (title, base_img, mask, cmap) in zip(axes, panels):
        ax.imshow(base_img)
        if mask is not None:
            ax.imshow(mask, alpha=0.35, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_dir / f"{patch.patch_id}_sensitivity.png", dpi=150)
    plt.close(fig)


def audit_width_calibration_sensitivity(
    *,
    summary_csv: str | Path,
    handoff_dir: str | Path,
    roads_gpkg: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    repo_root_path = repo_root()
    summary_path = resolve_path(
        summary_csv, repo_root_path=repo_root_path, prefer_repo=False
    ).resolve()
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
    summary_df = pd.read_csv(summary_path)
    roads_gdf = load_roads_layer(
        roads_gpkg_path, roads_layer=resolve_roads_layer_name(roads_gpkg_path)
    )
    patch_classes = compute_patch_class_presence(
        handoff_dir=handoff_dir_path, roads_gdf=roads_gdf
    )
    present_classes = sorted(
        {
            class_id
            for classes in patch_classes.values()
            for class_id in classes
            if class_id is not None
        }
    )
    final_widths: dict[int, int] = {}
    for row in summary_df.to_dict("records"):
        class_id = normalize_class(row.get("class"))
        final_value = row.get("final_width_px")
        if class_id is None or pd.isna(final_value):
            continue
        final_widths[int(class_id)] = int(final_value)
    audit_subset = select_audit_subset(patch_classes)
    audit_classes = sorted(
        {
            class_id
            for patch_id, _reason in audit_subset
            for class_id in patch_classes.get(patch_id, set())
        }
    )
    missing_classes = sorted(
        class_id for class_id in audit_classes if class_id not in final_widths
    )
    if missing_classes:
        raise ValueError(
            "Cannot run sensitivity audit because class widths are missing for: "
            f"{missing_classes}"
        )
    patch_map = load_patch_context_map(handoff_dir_path)
    cache: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    overlay_dir = out_dir_path / SENSITIVITY_OVERLAY_DIRNAME
    for patch_id, audit_reason in audit_subset:
        patch = patch_map[patch_id]
        patch_roads = roads_for_crs(roads_gdf, patch.crs_wkt, cache)
        class_width_tables = {
            "baseline": final_widths,
            "median_minus_1px": {
                class_id: max(1, int(width) - 1)
                for class_id, width in final_widths.items()
            },
            "median_plus_1px": {
                class_id: int(width) + 1 for class_id, width in final_widths.items()
            },
        }
        rendered: dict[str, np.ndarray] = {}
        counts: dict[str, dict[str, int]] = {}
        for variant, class_widths in class_width_tables.items():
            mask = render_mask_for_patch(
                patch, patch_roads, class_widths_px=class_widths
            )
            rendered[variant] = mask
            counts[variant] = {
                "foreground_pixels": int(np.count_nonzero(mask)),
                "connected_components": int(connected_components(mask)),
            }
        baseline_counts = counts["baseline"]
        for variant, variant_counts in counts.items():
            audit_rows.append(
                SensitivityRow(
                    patch_id=patch_id,
                    audit_reason=audit_reason,
                    variant=variant,
                    classes_present=",".join(
                        str(v) for v in sorted(patch_classes.get(patch_id, set()))
                    ),
                    foreground_pixels=int(variant_counts["foreground_pixels"]),
                    connected_components=int(variant_counts["connected_components"]),
                    delta_foreground_pixels_vs_baseline=int(
                        variant_counts["foreground_pixels"]
                        - baseline_counts["foreground_pixels"]
                    ),
                    delta_connected_components_vs_baseline=int(
                        variant_counts["connected_components"]
                        - baseline_counts["connected_components"]
                    ),
                ).to_row()
            )
        save_sensitivity_overlay(
            patch=patch,
            handoff_dir=handoff_dir_path,
            overlay_dir=overlay_dir,
            baseline_mask=rendered["baseline"],
            minus_mask=rendered["median_minus_1px"],
            plus_mask=rendered["median_plus_1px"],
            counts=counts,
        )
    sensitivity_df = pd.DataFrame(audit_rows, columns=SENSITIVITY_COLUMNS)
    sensitivity_path = out_dir_path / SENSITIVITY_FILENAME
    sensitivity_df.to_csv(sensitivity_path, index=False)
    return {
        "sensitivity_csv": str(sensitivity_path),
        "sensitivity_overlay_dir": str(overlay_dir),
        "present_classes": present_classes,
        "audit_subset": [patch_id for patch_id, _reason in audit_subset],
    }


__all__ = ["audit_width_calibration_sensitivity"]
