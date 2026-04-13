from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import GeometryCollection, LineString, MultiLineString, box

from dataselector.runtime.parameter_snapshot import compute_file_sha256

from .models import (
    COMMON_CLASS_QUOTAS,
    MANIFEST_FILENAME,
    MID_CLASS_QUOTAS,
    REPEAT_ALL_CLASSES,
    REPEAT_QUOTAS,
    SPECIAL_CLASS_QUOTAS,
    SUPPORTED_CLASSES,
    TASK_COLUMNS,
    TASK_FILENAME,
    WORKFLOW_VERSION,
    EligibilityParameters,
    PatchContext,
    RunManifest,
    TaskRecord,
    is_hamburg_patch,
    normalize_class,
    normalize_source_fid,
    repo_root,
    require_columns,
    resolve_path,
    utc_now,
    write_json,
)
from .runs import (
    maybe_archive_stale_prepare_run,
    maybe_sync_local_copy_from_source,
    prepare_sync_metadata_for_manifest,
    resolve_roads_layer_name,
)


def load_selected_patches(
    handoff_dir: Path, *, exclude_hamburg: bool = True
) -> pd.DataFrame:
    selected_path = handoff_dir / "selected_patches.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected_patches.csv: {selected_path}")
    df = pd.read_csv(selected_path)
    require_columns(
        df,
        [
            "patch_id",
            "tile_shortname",
            "selection_rank",
            "patch_index",
            "quicklook_path",
        ],
        "selected_patches.csv",
    )
    df = df.copy()
    df["patch_id"] = df["patch_id"].astype(str)
    df["tile_shortname"] = df["tile_shortname"].astype(str)
    df["selection_rank"] = (
        pd.to_numeric(df["selection_rank"], errors="coerce").fillna(0).astype(int)
    )
    df["patch_index"] = (
        pd.to_numeric(df["patch_index"], errors="coerce").fillna(0).astype(int)
    )
    df = df.sort_values(
        ["selection_rank", "patch_index", "patch_id"], kind="stable"
    ).reset_index(drop=True)
    df["_patch_order"] = np.arange(len(df), dtype=int)
    if not exclude_hamburg:
        return df
    filtered = (
        df.loc[
            ~df.apply(
                lambda row: is_hamburg_patch(
                    str(row["patch_id"]), str(row["tile_shortname"])
                ),
                axis=1,
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
    filtered["_patch_order"] = np.arange(len(filtered), dtype=int)
    return filtered


def build_patch_contexts(
    handoff_dir: Path, selected_df: pd.DataFrame
) -> list[PatchContext]:
    contexts: list[PatchContext] = []
    for row in selected_df.to_dict("records"):
        quicklook_relpath = str(row["quicklook_path"]).strip()
        quicklook_path = (handoff_dir / quicklook_relpath).resolve()
        if not quicklook_path.exists():
            raise FileNotFoundError(
                f"Missing quicklook for patch {row['patch_id']}: {quicklook_path}"
            )
        with rasterio.open(quicklook_path) as ds:
            contexts.append(
                PatchContext(
                    patch_id=str(row["patch_id"]),
                    tile_shortname=str(row["tile_shortname"]),
                    quicklook_relpath=quicklook_relpath,
                    quicklook_path=quicklook_path,
                    width=int(ds.width),
                    height=int(ds.height),
                    transform=ds.transform,
                    crs_wkt=str(ds.crs) if ds.crs is not None else "",
                    pixel_size=float(np.mean([abs(ds.res[0]), abs(ds.res[1])])),
                    bounds=(
                        float(ds.bounds.left),
                        float(ds.bounds.bottom),
                        float(ds.bounds.right),
                        float(ds.bounds.top),
                    ),
                    selection_rank=int(row["selection_rank"]),
                    patch_index=int(row["patch_index"]),
                    patch_order=int(row["_patch_order"]),
                )
            )
    return contexts


def load_roads_layer(
    roads_gpkg: Path,
    *,
    roads_layer: str,
) -> Any:
    import pyogrio

    if not roads_gpkg.exists():
        raise FileNotFoundError(f"Road source not found: {roads_gpkg}")
    gdf = pyogrio.read_dataframe(roads_gpkg, layer=roads_layer, fid_as_index=True)
    if gdf.empty:
        raise ValueError(f"Road layer is empty: {roads_gpkg} [{roads_layer}]")
    if "class" not in gdf.columns:
        raise ValueError(f"Road layer missing required 'class' field: {roads_gpkg}")
    fid_column = str(gdf.index.name or "fid")
    gdf = gdf.reset_index(drop=False).rename(columns={fid_column: "source_fid"}).copy()
    gdf = gdf.reset_index(drop=False).rename(columns={"index": "_source_row"}).copy()
    gdf["class"] = gdf["class"].map(normalize_class)
    gdf = gdf.loc[gdf["class"].isin(SUPPORTED_CLASSES)].copy()
    gdf["source_fid"] = gdf["source_fid"].map(normalize_source_fid)
    gdf["source_feature_id"] = gdf["_source_row"].map(
        lambda value: f"row_{int(value):06d}"
    )
    gdf = gdf.loc[gdf.geometry.notna()].copy()
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    return gdf


def source_fid_lookup_from_roads(
    roads_gpkg: Path,
    *,
    roads_layer: str | None = None,
) -> dict[str, str]:
    if not roads_gpkg.exists():
        return {}
    layer_name = roads_layer or resolve_roads_layer_name(roads_gpkg)
    roads_gdf = load_roads_layer(roads_gpkg, roads_layer=layer_name)
    rows = roads_gdf[["source_feature_id", "source_fid"]].to_dict("records")
    return {
        str(row["source_feature_id"]): normalize_source_fid(row.get("source_fid"))
        for row in rows
        if str(row.get("source_feature_id", "")).strip()
    }


def roads_for_crs(roads_gdf: Any, crs_wkt: str, cache: dict[str, Any]) -> Any:
    if crs_wkt in cache:
        return cache[crs_wkt]
    if not crs_wkt or roads_gdf.crs is None:
        cache[crs_wkt] = roads_gdf
        return roads_gdf
    cache[crs_wkt] = roads_gdf.to_crs(crs_wkt)
    return cache[crs_wkt]


def subset_by_bounds(gdf: Any, bounds: tuple[float, float, float, float]) -> Any:
    minx, miny, maxx, maxy = bounds
    try:
        sindex = gdf.sindex
    except Exception:
        sindex = None
    if sindex is not None:
        idx = list(sindex.intersection((minx, miny, maxx, maxy)))
        if idx:
            return gdf.iloc[idx].copy()
        return gdf.iloc[0:0].copy()
    return gdf.cx[minx:maxx, miny:maxy].copy()


def extract_lines(geometry: Any) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return [
            part
            for part in geometry.geoms
            if isinstance(part, LineString) and not part.is_empty
        ]
    if isinstance(geometry, GeometryCollection):
        out: list[LineString] = []
        for geom in geometry.geoms:
            out.extend(extract_lines(geom))
        return out
    return []


def pixel_window(
    *,
    anchor_x_px: int,
    anchor_y_px: int,
    crop_size_px: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    half = crop_size_px // 2
    x0 = anchor_x_px - half
    y0 = anchor_y_px - half
    x1 = x0 + crop_size_px
    y1 = y0 + crop_size_px
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return x0, y0, x1, y1


def pixel_window_to_world_box(transform: Any, window: tuple[int, int, int, int]) -> Any:
    x0, y0, x1, y1 = window
    corners = [
        transform * (x0, y0),
        transform * (x1, y0),
        transform * (x1, y1),
        transform * (x0, y1),
    ]
    xs = [float(pt[0]) for pt in corners]
    ys = [float(pt[1]) for pt in corners]
    return box(min(xs), min(ys), max(xs), max(ys))


def anchor_distance_along_line(
    *,
    line_length: float,
    endpoint_exclusion_fraction: float,
    anchor_fraction: float,
) -> float | None:
    start = endpoint_exclusion_fraction * line_length
    end = (1.0 - endpoint_exclusion_fraction) * line_length
    if end <= start:
        return None
    return start + anchor_fraction * (end - start)


def candidate_rows_for_patch(
    patch: PatchContext,
    roads_gdf: Any,
    *,
    crop_size_px: int,
    eligibility: EligibilityParameters,
) -> list[dict[str, Any]]:
    patch_poly = box(*patch.bounds)
    subset = subset_by_bounds(roads_gdf, patch.bounds)
    if subset.empty:
        return []
    margin_px = eligibility.border_margin_px(crop_size_px)
    out: list[dict[str, Any]] = []
    for row in subset.sort_values(["class", "source_feature_id"]).to_dict("records"):
        class_id = normalize_class(row.get("class"))
        if class_id not in SUPPORTED_CLASSES:
            continue
        clipped = row["geometry"].intersection(patch_poly)
        if clipped.is_empty:
            continue
        for part_idx, line in enumerate(extract_lines(clipped)):
            if line.length <= 0:
                continue
            seen_pixels: set[tuple[int, int]] = set()
            for anchor_idx, anchor_fraction in enumerate(eligibility.anchor_positions):
                distance = anchor_distance_along_line(
                    line_length=float(line.length),
                    endpoint_exclusion_fraction=eligibility.endpoint_exclusion_fraction,
                    anchor_fraction=float(anchor_fraction),
                )
                if distance is None:
                    continue
                point = line.interpolate(distance)
                px, py = ~patch.transform * (float(point.x), float(point.y))
                anchor_x_px = int(round(float(px)))
                anchor_y_px = int(round(float(py)))
                if (anchor_x_px, anchor_y_px) in seen_pixels:
                    continue
                if (
                    anchor_x_px < margin_px
                    or anchor_y_px < margin_px
                    or anchor_x_px > (patch.width - margin_px)
                    or anchor_y_px > (patch.height - margin_px)
                ):
                    continue
                window = pixel_window(
                    anchor_x_px=anchor_x_px,
                    anchor_y_px=anchor_y_px,
                    crop_size_px=crop_size_px,
                    width=patch.width,
                    height=patch.height,
                )
                if window is None:
                    continue
                crop_poly = pixel_window_to_world_box(patch.transform, window)
                line_support_px = float(line.intersection(crop_poly).length) / max(
                    patch.pixel_size, 1e-9
                )
                if line_support_px < float(eligibility.minimum_in_crop_line_support_px):
                    continue
                seen_pixels.add((anchor_x_px, anchor_y_px))
                source_feature_id = str(row["source_feature_id"])
                out.append(
                    {
                        "candidate_id": (
                            f"{patch.patch_id}__{source_feature_id}"
                            f"__part{part_idx:02d}__anchor{anchor_idx:02d}"
                        ),
                        "class": class_id,
                        "patch_id": patch.patch_id,
                        "tile_shortname": patch.tile_shortname,
                        "source_fid": normalize_source_fid(row.get("source_fid")),
                        "source_feature_id": source_feature_id,
                        "quicklook_path": patch.quicklook_relpath,
                        "anchor_x_px": anchor_x_px,
                        "anchor_y_px": anchor_y_px,
                        "crop_size_px": int(crop_size_px),
                        "patch_order": patch.patch_order,
                        "selection_rank": patch.selection_rank,
                        "patch_index": patch.patch_index,
                        "part_index": part_idx,
                        "line_support_px": round(line_support_px, 6),
                    }
                )
    return out


def build_candidate_dataframe(
    *,
    handoff_dir: Path,
    roads_gdf: Any,
    crop_size_px: int,
    eligibility: EligibilityParameters,
) -> tuple[pd.DataFrame, list[str]]:
    selected_df = load_selected_patches(handoff_dir)
    contexts = build_patch_contexts(handoff_dir, selected_df)
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    excluded_patch_ids = sorted(
        {
            str(row["patch_id"])
            for row in pd.read_csv(handoff_dir / "selected_patches.csv").to_dict(
                "records"
            )
            if is_hamburg_patch(
                str(row.get("patch_id", "")), str(row.get("tile_shortname", ""))
            )
        }
    )
    for patch in contexts:
        patch_roads = roads_for_crs(roads_gdf, patch.crs_wkt, cache)
        rows.extend(
            candidate_rows_for_patch(
                patch,
                patch_roads,
                crop_size_px=crop_size_px,
                eligibility=eligibility,
            )
        )
    columns = [
        "candidate_id",
        "class",
        "patch_id",
        "tile_shortname",
        "source_fid",
        "source_feature_id",
        "quicklook_path",
        "anchor_x_px",
        "anchor_y_px",
        "crop_size_px",
        "patch_order",
        "selection_rank",
        "patch_index",
        "part_index",
        "line_support_px",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), excluded_patch_ids
    df = pd.DataFrame(rows)
    df["source_fid"] = df["source_fid"].map(normalize_source_fid)
    df = df.sort_values(
        [
            "class",
            "patch_order",
            "tile_shortname",
            "patch_id",
            "source_feature_id",
            "part_index",
            "anchor_x_px",
            "anchor_y_px",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return df, excluded_patch_ids


def rng_for(seed: int, *parts: Any) -> np.random.Generator:
    value = int(seed)
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**32 - 1)
    return np.random.default_rng(value)


def shuffle_records(
    records: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    if len(records) <= 1:
        return list(records)
    order = rng.permutation(len(records))
    return [records[int(idx)] for idx in order]


def round_robin_select(
    group_df: pd.DataFrame,
    *,
    target: int,
    seed: int,
    tile_target: int | None = None,
) -> list[dict[str, Any]]:
    if group_df.empty:
        return []
    rng = rng_for(seed, "round_robin", group_df["class"].iloc[0])
    tiles = list(dict.fromkeys(group_df["tile_shortname"].astype(str).tolist()))
    if len(tiles) > 1:
        tiles = list(rng.permutation(tiles))
    base_tiles = tiles[: min(len(tiles), int(tile_target))] if tile_target else tiles
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for tile in tiles:
        tile_records = group_df.loc[group_df["tile_shortname"] == tile].to_dict(
            "records"
        )
        buckets[tile] = deque(shuffle_records(tile_records, rng_for(seed, tile)))
    active_tiles = list(base_tiles)
    extra_tiles = [tile for tile in tiles if tile not in active_tiles]
    selected: list[dict[str, Any]] = []
    while len(selected) < target and active_tiles:
        made_progress = False
        for tile in list(active_tiles):
            if len(selected) >= target:
                break
            if buckets[tile]:
                selected.append(buckets[tile].popleft())
                made_progress = True
        if len(selected) >= target:
            break
        if not made_progress and extra_tiles:
            active_tiles.append(extra_tiles.pop(0))
            made_progress = True
        if not made_progress:
            break
    return selected


def select_primary_tasks(candidates_df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_id in SUPPORTED_CLASSES:
        group = candidates_df.loc[candidates_df["class"] == class_id].copy()
        if group.empty:
            continue
        if class_id in REPEAT_ALL_CLASSES or class_id == 8:
            selected = group.to_dict("records")
        elif class_id in COMMON_CLASS_QUOTAS:
            selected = round_robin_select(
                group, target=COMMON_CLASS_QUOTAS[class_id], seed=seed, tile_target=4
            )
        elif class_id in MID_CLASS_QUOTAS:
            selected = round_robin_select(
                group, target=MID_CLASS_QUOTAS[class_id], seed=seed, tile_target=3
            )
        elif class_id in SPECIAL_CLASS_QUOTAS:
            selected = round_robin_select(
                group,
                target=SPECIAL_CLASS_QUOTAS[class_id],
                seed=seed,
                tile_target=len(group["tile_shortname"].unique()),
            )
        else:
            selected = group.to_dict("records")
        rows.extend(selected)
    if not rows:
        return pd.DataFrame(columns=TASK_COLUMNS)
    primary_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["candidate_id"])
        .reset_index(drop=True)
    )
    shuffled = shuffle_records(
        primary_df.to_dict("records"), rng_for(seed, "primary_queue")
    )
    task_rows = [
        TaskRecord(
            task_id=f"task_{idx:05d}",
            candidate_id=str(row["candidate_id"]),
            class_id=int(row["class"]),
            patch_id=str(row["patch_id"]),
            tile_shortname=str(row["tile_shortname"]),
            source_fid=normalize_source_fid(row.get("source_fid")),
            source_feature_id=str(row["source_feature_id"]),
            quicklook_path=str(row["quicklook_path"]),
            anchor_x_px=int(row["anchor_x_px"]),
            anchor_y_px=int(row["anchor_y_px"]),
            crop_size_px=int(row["crop_size_px"]),
            queue_position=idx,
            pass_type="primary",
            repeat_of_task_id="",
        ).to_row()
        for idx, row in enumerate(shuffled, start=1)
    ]
    return pd.DataFrame(task_rows, columns=TASK_COLUMNS)


def select_repeat_tasks(primary_df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if primary_df.empty:
        return pd.DataFrame(columns=TASK_COLUMNS)
    selected_repeat_rows: list[dict[str, Any]] = []
    for class_id in sorted(set(primary_df["class"].astype(int).tolist())):
        group = primary_df.loc[primary_df["class"] == class_id].copy()
        if group.empty:
            continue
        if class_id in REPEAT_ALL_CLASSES:
            repeat_source = group.to_dict("records")
        else:
            quota = int(REPEAT_QUOTAS.get(class_id, 0))
            if quota <= 0:
                continue
            repeat_source = shuffle_records(
                group.to_dict("records"), rng_for(seed, "repeat", class_id)
            )[: min(quota, len(group))]
        selected_repeat_rows.extend(repeat_source)
    shuffled = shuffle_records(selected_repeat_rows, rng_for(seed, "repeat_queue"))
    start_idx = len(primary_df) + 1
    repeat_rows = [
        TaskRecord(
            task_id=f"task_{start_idx + offset:05d}",
            candidate_id=str(row["candidate_id"]),
            class_id=int(row["class"]),
            patch_id=str(row["patch_id"]),
            tile_shortname=str(row["tile_shortname"]),
            source_fid=normalize_source_fid(row.get("source_fid")),
            source_feature_id=str(row["source_feature_id"]),
            quicklook_path=str(row["quicklook_path"]),
            anchor_x_px=int(row["anchor_x_px"]),
            anchor_y_px=int(row["anchor_y_px"]),
            crop_size_px=int(row["crop_size_px"]),
            queue_position=start_idx + offset,
            pass_type="repeat",
            repeat_of_task_id=str(row["task_id"]),
        ).to_row()
        for offset, row in enumerate(shuffled, start=0)
    ]
    return pd.DataFrame(repeat_rows, columns=TASK_COLUMNS)


def prepare_width_calibration(
    *,
    handoff_dir: str | Path,
    roads_gpkg: str | Path,
    roads_layer: str,
    seed: int,
    crop_size_px: int,
    out_dir: str | Path,
    prompt_for_sync: bool = False,
) -> dict[str, Any]:
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
    eligibility = EligibilityParameters()
    current_local_sha = compute_file_sha256(roads_gpkg_path)
    current_local_sha, _ = maybe_sync_local_copy_from_source(
        roads_gpkg_path,
        roads_layer=roads_layer,
        current_local_sha=current_local_sha,
        prompt_for_sync=prompt_for_sync,
    )
    sync_metadata_path, sync_metadata = prepare_sync_metadata_for_manifest(
        roads_gpkg_path,
        roads_gpkg_sha256=current_local_sha,
    )
    archived_run_dir = maybe_archive_stale_prepare_run(
        out_dir_path,
        handoff_dir=handoff_dir_path,
        roads_layer=roads_layer,
        seed=seed,
        crop_size_px=crop_size_px,
        current_local_sha=current_local_sha,
        current_sync_metadata=sync_metadata,
        prompt_for_archive=prompt_for_sync,
    )
    roads_gdf = load_roads_layer(roads_gpkg_path, roads_layer=roads_layer)
    candidates_df, excluded_patch_ids = build_candidate_dataframe(
        handoff_dir=handoff_dir_path,
        roads_gdf=roads_gdf,
        crop_size_px=int(crop_size_px),
        eligibility=eligibility,
    )
    primary_df = select_primary_tasks(candidates_df, seed=int(seed))
    repeat_df = select_repeat_tasks(primary_df, seed=int(seed))
    tasks_df = pd.concat([primary_df, repeat_df], ignore_index=True)
    if not tasks_df.empty:
        tasks_df = tasks_df[TASK_COLUMNS].copy()
    tasks_path = out_dir_path / TASK_FILENAME
    tasks_df.to_csv(tasks_path, index=False)
    selected_csv = handoff_dir_path / "selected_patches.csv"
    extras = {
        "supported_classes": list(SUPPORTED_CLASSES),
        "class_quotas": {
            "common": COMMON_CLASS_QUOTAS,
            "mid_frequency": MID_CLASS_QUOTAS,
            "special": SPECIAL_CLASS_QUOTAS,
        },
        "repeat_schedule": {
            "repeat_all_classes": sorted(REPEAT_ALL_CLASSES),
            "quota_classes": REPEAT_QUOTAS,
        },
        "eligibility_parameters": {
            **asdict(eligibility),
            "minimum_border_margin_px": int(
                eligibility.border_margin_px(int(crop_size_px))
            ),
        },
        "selected_patches_csv": str(selected_csv),
        "selected_patches_csv_sha256": compute_file_sha256(selected_csv),
        "excluded_patch_ids": excluded_patch_ids,
        "candidate_count": int(len(candidates_df)),
        "primary_task_count": int(len(primary_df)),
        "repeat_task_count": int(len(repeat_df)),
        "tasks_csv_path": str(tasks_path),
        "tasks_csv_sha256": compute_file_sha256(tasks_path),
        "hamburg_excluded_at_task_generation": True,
        "archived_previous_run": archived_run_dir is not None,
    }
    if sync_metadata_path is not None and sync_metadata is not None:
        extras["sync_metadata_path"] = str(sync_metadata_path)
        extras["sync_source_gpkg"] = str(sync_metadata.source_gpkg_path)
        extras["sync_source_gpkg_sha256"] = str(sync_metadata.source_gpkg_sha256)
    manifest = RunManifest(
        workflow_version=WORKFLOW_VERSION,
        generated_utc=utc_now(),
        handoff_dir=str(handoff_dir_path),
        roads_gpkg=str(roads_gpkg_path),
        roads_gpkg_sha256=current_local_sha,
        roads_layer=str(roads_layer),
        seed=int(seed),
        crop_size_px=int(crop_size_px),
        extras=extras,
    )
    manifest_path = out_dir_path / MANIFEST_FILENAME
    write_json(manifest_path, manifest.to_payload())
    return {
        "handoff_dir": str(handoff_dir_path),
        "roads_gpkg": str(roads_gpkg_path),
        "tasks_csv": str(tasks_path),
        "manifest_json": str(manifest_path),
        "candidate_count": int(len(candidates_df)),
        "primary_task_count": int(len(primary_df)),
        "repeat_task_count": int(len(repeat_df)),
        "excluded_patch_ids": excluded_patch_ids,
        "archived_previous_run": archived_run_dir is not None,
        "archive_dir": str(archived_run_dir) if archived_run_dir is not None else "",
    }


__all__ = [
    "anchor_distance_along_line",
    "build_candidate_dataframe",
    "build_patch_contexts",
    "candidate_rows_for_patch",
    "extract_lines",
    "load_roads_layer",
    "load_selected_patches",
    "pixel_window",
    "pixel_window_to_world_box",
    "prepare_width_calibration",
    "resolve_roads_layer_name",
    "roads_for_crs",
    "select_primary_tasks",
    "select_repeat_tasks",
    "shuffle_records",
    "source_fid_lookup_from_roads",
    "subset_by_bounds",
]
