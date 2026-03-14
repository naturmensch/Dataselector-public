"""Build deterministic patch-level annotation plans from thesis run selections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from sklearn.model_selection import GroupKFold

from dataselector.cli_decorators import cli_command
from dataselector.runtime.parameter_snapshot import compute_file_sha256

DEFAULT_PRIMARY_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.33, 0.33),
    (0.67, 0.67),
)
DEFAULT_FALLBACK_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.25, 0.75),
    (0.75, 0.25),
    (0.50, 0.25),
    (0.50, 0.75),
)


@dataclass(frozen=True)
class PatchQCDecision:
    passed: bool
    reason: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class PatchWindow:
    x0: int
    y0: int
    x1: int
    y1: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_bool_flag(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_anchor_string(raw: str) -> list[tuple[float, float]]:
    anchors: list[tuple[float, float]] = []
    for token in [part.strip() for part in raw.split(";") if part.strip()]:
        bits = [part.strip() for part in token.split(",")]
        if len(bits) != 2:
            raise ValueError(
                "Invalid anchor token "
                f"{token!r}; expected format 'x,y;x,y' with normalized coordinates."
            )
        x = float(bits[0])
        y = float(bits[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Anchor {token!r} outside normalized range [0, 1].")
        anchors.append((x, y))
    if not anchors:
        raise ValueError("Fallback anchors must not be empty.")
    return anchors


def _default_primary_anchors(patches_per_tile: int) -> list[tuple[float, float]]:
    if patches_per_tile <= len(DEFAULT_PRIMARY_ANCHORS):
        return list(DEFAULT_PRIMARY_ANCHORS[:patches_per_tile])

    extra = [
        (0.33, 0.67),
        (0.67, 0.33),
        (0.50, 0.50),
        (0.20, 0.20),
        (0.80, 0.80),
        (0.20, 0.80),
        (0.80, 0.20),
    ]
    out = list(DEFAULT_PRIMARY_ANCHORS)
    for anchor in extra:
        if len(out) >= patches_per_tile:
            break
        out.append(anchor)
    while len(out) < patches_per_tile:
        out.append((0.50, 0.50))
    return out[:patches_per_tile]


def _resolve_image_path(path_value: str | Path, *, run_dir: Path) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    run_relative = run_dir / raw
    if run_relative.exists():
        return run_relative
    repo_relative = _repo_root() / raw
    if repo_relative.exists():
        return repo_relative
    cwd_relative = Path.cwd() / raw
    if cwd_relative.exists():
        return cwd_relative
    return run_relative


def _rel_to(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except Exception:
        return str(target.resolve())


def _required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def _select_input_rows(
    *, run_dir: Path, include_case: bool
) -> tuple[pd.DataFrame, dict[str, Path]]:
    core_path = _required_file(run_dir / "selection_core.csv", "selection_core.csv")
    case_path = _required_file(run_dir / "selection_case.csv", "selection_case.csv")
    final_path = _required_file(
        run_dir / "selection_final_with_cases.csv", "selection_final_with_cases.csv"
    )
    contract_path = _required_file(
        run_dir / "selection_contract.json", "selection_contract.json"
    )
    metadata_path = _required_file(run_dir / "run_metadata.json", "run_metadata.json")

    core_df = pd.read_csv(core_path)
    case_df = pd.read_csv(case_path)
    final_df = pd.read_csv(final_path)

    required_cols = {"shortName", "image_path", "selection_rank"}
    missing = required_cols - set(final_df.columns)
    if missing:
        raise ValueError(
            "selection_final_with_cases.csv missing required columns: "
            f"{sorted(missing)}"
        )

    use_df = final_df.copy() if include_case else core_df.copy()
    if use_df.empty:
        raise ValueError("Selection input is empty; cannot build annotation plan.")

    use_df["shortName"] = use_df["shortName"].astype(str)
    use_df["selection_rank"] = pd.to_numeric(use_df["selection_rank"], errors="coerce")
    use_df = use_df.sort_values("selection_rank", kind="stable").reset_index(drop=True)

    case_shortnames = set(case_df["shortName"].astype(str).tolist())
    use_df["selection_group"] = use_df["shortName"].map(
        lambda name: "case" if name in case_shortnames else "core"
    )

    source_files = {
        "selection_core": core_path,
        "selection_case": case_path,
        "selection_final_with_cases": final_path,
        "selection_contract": contract_path,
        "run_metadata": metadata_path,
    }
    return use_df, source_files


def _compute_patch_window(
    *, width: int, height: int, patch_size: int, anchor: tuple[float, float]
) -> PatchWindow:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Tile smaller than patch_size ({width}x{height} < {patch_size}x{patch_size})"
        )

    center_x = int(round(float(anchor[0]) * float(width - 1)))
    center_y = int(round(float(anchor[1]) * float(height - 1)))

    x0 = center_x - patch_size // 2
    y0 = center_y - patch_size // 2

    x0 = max(0, min(x0, width - patch_size))
    y0 = max(0, min(y0, height - patch_size))
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    return PatchWindow(x0=x0, y0=y0, x1=x1, y1=y1)


def _evaluate_patch_qc(patch_img: Image.Image, *, qc_mode: str) -> PatchQCDecision:
    mode = str(qc_mode).strip().lower()
    if mode == "none":
        return PatchQCDecision(passed=True, reason=None, metrics={})
    if mode != "heuristic_v1":
        raise ValueError(f"Unsupported qc_mode: {qc_mode!r}")

    rgb = np.asarray(patch_img.convert("RGB"), dtype=np.uint8)
    gray = rgb.mean(axis=2)

    std_val = float(np.std(gray))
    dynamic_range = float(np.max(gray) - np.min(gray))
    near_white = float(np.mean(gray >= 245.0))
    near_black = float(np.mean(gray <= 10.0))

    h, w = gray.shape
    edge_band = max(8, int(min(h, w) * 0.12))
    edge_mask = np.zeros_like(gray, dtype=bool)
    edge_mask[:edge_band, :] = True
    edge_mask[-edge_band:, :] = True
    edge_mask[:, :edge_band] = True
    edge_mask[:, -edge_band:] = True

    center_margin = max(8, int(min(h, w) * 0.2))
    center_view = gray[
        center_margin : max(center_margin + 1, h - center_margin),
        center_margin : max(center_margin + 1, w - center_margin),
    ]
    edge_dark = float(np.mean(gray[edge_mask] <= 70.0))
    center_dark = float(np.mean(center_view <= 70.0))

    metrics = {
        "std_gray": std_val,
        "dynamic_range": dynamic_range,
        "near_white": near_white,
        "near_black": near_black,
        "edge_dark": edge_dark,
        "center_dark": center_dark,
    }

    if dynamic_range < 6.0 or std_val < 2.0:
        return PatchQCDecision(False, "blank_low_variance", metrics)
    if near_white > 0.995 or near_black > 0.995:
        return PatchQCDecision(False, "no_data_uniform", metrics)
    if edge_dark > 0.22 and center_dark < 0.04:
        return PatchQCDecision(False, "legend_dominant_edge", metrics)

    return PatchQCDecision(True, None, metrics)


def _load_source_georeference(
    *, source_image_path: Path
) -> tuple[Any, Affine, int, int]:
    try:
        with rasterio.open(source_image_path) as src:
            source_crs = src.crs
            source_transform = src.transform
            width = int(src.width)
            height = int(src.height)
    except Exception as exc:
        raise ValueError(
            f"Failed to read georeference for source image: {source_image_path}"
        ) from exc

    if source_crs is None:
        raise ValueError(
            "Missing CRS georeference for source image "
            f"(requires embedded georef or sidecar): {source_image_path}"
        )
    if source_transform is None or source_transform.is_identity:
        raise ValueError(
            "Missing/non-georeferenced affine transform for source image "
            f"(requires embedded georef or sidecar): {source_image_path}"
        )

    coeffs = tuple(source_transform)[:6]
    if not all(np.isfinite(value) for value in coeffs):
        raise ValueError(
            "Invalid affine transform coefficients for source image: "
            f"{source_image_path}"
        )

    return source_crs, source_transform, width, height


def _geotiff_block_size(*, width: int, height: int) -> int:
    edge = min(int(width), int(height), 256)
    block = edge - (edge % 16)
    if block < 16:
        return 0
    return block


def _write_patch_quicklook_geotiff(
    *,
    patch_img: Image.Image,
    source_crs: Any,
    source_transform: Affine,
    window: PatchWindow,
    output_path: Path,
) -> None:
    rgb_patch = patch_img.convert("RGB")
    arr_hwc = np.asarray(rgb_patch, dtype=np.uint8)
    if arr_hwc.ndim != 3 or arr_hwc.shape[2] != 3:
        raise ValueError(f"Expected RGB patch image for quicklook: {output_path}")

    height, width = int(arr_hwc.shape[0]), int(arr_hwc.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid patch shape for quicklook: {output_path}")

    patch_window = Window(
        col_off=float(window.x0),
        row_off=float(window.y0),
        width=float(window.x1 - window.x0),
        height=float(window.y1 - window.y0),
    )
    patch_transform = window_transform(patch_window, source_transform)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "crs": source_crs,
        "transform": patch_transform,
        "compress": "DEFLATE",
        "predictor": 2,
        "interleave": "pixel",
    }

    block_size = _geotiff_block_size(width=width, height=height)
    if block_size > 0:
        profile.update(
            {
                "tiled": True,
                "blockxsize": block_size,
                "blockysize": block_size,
            }
        )
    else:
        profile["tiled"] = False

    arr_chw = np.transpose(arr_hwc, (2, 0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr_chw)


def _write_patch_manifest_json(
    *,
    out_path: Path,
    records: list[dict[str, Any]],
    run_id: str,
    patch_size: int,
    patches_per_tile: int,
    include_case: bool,
    qc_mode: str,
    fallback_anchors: list[tuple[float, float]],
) -> None:
    payload = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "patch_size_px": int(patch_size),
        "patches_per_tile": int(patches_per_tile),
        "include_case": bool(include_case),
        "qc_mode": qc_mode,
        "fallback_anchors": [list(anchor) for anchor in fallback_anchors],
        "records": records,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _build_split_manifest(
    *,
    manifest_df: pd.DataFrame,
    run_id: str,
    output_path: Path,
    n_splits: int,
) -> tuple[pd.DataFrame, str]:
    passed_df = manifest_df[manifest_df["qc_status"] == "qc_passed"].copy()
    if passed_df.empty:
        raise ValueError("No qc_passed patches available; cannot build split manifest.")

    passed_df = passed_df.sort_values(
        ["tile_shortname", "patch_index", "patch_id"], kind="stable"
    ).reset_index(drop=True)

    groups = passed_df["tile_shortname"].astype(str).to_numpy()
    unique_tiles = np.unique(groups)
    if len(unique_tiles) < n_splits:
        raise ValueError(
            "GroupKFold requires at least n_splits unique tiles; "
            f"got {len(unique_tiles)} unique tiles for n_splits={n_splits}."
        )

    gkf = GroupKFold(n_splits=n_splits)
    fold_by_patch: dict[str, int] = {}
    dummy_x = np.zeros(len(passed_df), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
        gkf.split(dummy_x, groups=groups), start=1
    ):
        for row_idx in test_idx.tolist():
            patch_id = str(passed_df.iloc[row_idx]["patch_id"])
            fold_by_patch[patch_id] = int(fold_idx)

    # Leakage guard: all patches of one tile must share one fold.
    tile_to_folds: dict[str, set[int]] = {}
    for patch_id, fold_id in fold_by_patch.items():
        tile_name = str(
            passed_df.loc[passed_df["patch_id"] == patch_id, "tile_shortname"].iloc[0]
        )
        tile_to_folds.setdefault(tile_name, set()).add(int(fold_id))
    leakage = {tile: folds for tile, folds in tile_to_folds.items() if len(folds) > 1}
    if leakage:
        raise RuntimeError(f"Tile leakage detected in GroupKFold assignment: {leakage}")

    manifest_df = manifest_df.copy()
    manifest_df["split_fold"] = (
        manifest_df["patch_id"].map(fold_by_patch).astype("Int64")
    )

    fold_entries: list[dict[str, Any]] = []
    for fold_id in range(1, n_splits + 1):
        fold_rows = passed_df[passed_df["patch_id"].map(fold_by_patch.get) == fold_id]
        patch_ids = fold_rows["patch_id"].astype(str).tolist()
        tile_ids = sorted(set(fold_rows["tile_shortname"].astype(str).tolist()))
        fold_entries.append(
            {
                "fold": int(fold_id),
                "n_patches": int(len(patch_ids)),
                "n_tiles": int(len(tile_ids)),
                "patch_ids": patch_ids,
                "tile_shortnames": tile_ids,
            }
        )

    payload = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "grouping_key": "tile_shortname",
        "splitter": "GroupKFold",
        "n_splits": int(n_splits),
        "counts": {
            "total_patches": int(len(manifest_df)),
            "qc_passed_patches": int((manifest_df["qc_status"] == "qc_passed").sum()),
            "qc_rejected_patches": int((manifest_df["qc_status"] != "qc_passed").sum()),
            "unique_tiles": int(manifest_df["tile_shortname"].nunique()),
        },
        "folds": fold_entries,
        "patch_to_fold": {
            str(row["patch_id"]): int(row["split_fold"])
            for _, row in manifest_df.dropna(subset=["split_fold"]).iterrows()
        },
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    split_sha = compute_file_sha256(output_path)
    payload["split_manifest_sha256"] = split_sha
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return manifest_df, split_sha


def _write_annotation_scaffolding(
    *,
    output_dir: Path,
    manifest_df: pd.DataFrame,
) -> dict[str, Path]:
    class_mapping_path = output_dir / "class_mapping.yaml"
    class_mapping_payload = {
        "version": 1,
        "primary_target": "road_union",
        "notes": (
            "Set `qgis_classes` to your real annotation class names. "
            "`road_union` remains the primary thesis endpoint."
        ),
        "qgis_classes": [],
        "derived_targets": {
            "road_union": {
                "type": "union",
                "source_classes": [],
            }
        },
    }
    class_mapping_path.write_text(
        yaml.safe_dump(class_mapping_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    progress_path = output_dir / "annotation_progress.csv"
    progress_df = manifest_df[manifest_df["qc_status"] == "qc_passed"][
        ["patch_id", "tile_shortname", "split_fold"]
    ].copy()
    progress_df["status"] = "pending"
    progress_df["annotator"] = ""
    progress_df["started_at_utc"] = ""
    progress_df["finished_at_utc"] = ""
    progress_df["notes"] = ""
    progress_df.to_csv(progress_path, index=False)

    qa_path = output_dir / "annotation_qa_log.csv"
    pd.DataFrame(
        columns=[
            "patch_id",
            "qa_status",
            "qa_reviewer",
            "qa_timestamp_utc",
            "issue_type",
            "issue_detail",
            "resolution",
        ]
    ).to_csv(qa_path, index=False)

    return {
        "class_mapping": class_mapping_path,
        "annotation_progress": progress_path,
        "annotation_qa_log": qa_path,
    }


def run_thesis_build_annotation_plan(
    *,
    run_dir: str | Path,
    patch_size: int = 1024,
    patches_per_tile: int = 2,
    include_case: bool = True,
    fallback_anchors: str = "0.25,0.75;0.75,0.25;0.50,0.25;0.50,0.75",
    qc_mode: str = "heuristic_v1",
    output_subdir: str = "annotation_plan",
    split_n_splits: int = 5,
    dataset_version: str = "annotation_plan_v1",
    patch_policy_version: str = "kdr_patch_policy_v1",
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if patches_per_tile <= 0:
        raise ValueError("patches_per_tile must be > 0")
    if split_n_splits < 2:
        raise ValueError("split_n_splits must be >= 2")

    selection_df, source_files = _select_input_rows(
        run_dir=run_path, include_case=include_case
    )
    fallback_anchor_list = _parse_anchor_string(fallback_anchors)
    primary_anchors = _default_primary_anchors(patches_per_tile)

    out_dir = run_path / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    quicklook_dir = out_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        key: compute_file_sha256(path) for key, path in source_files.items()
    }

    records: list[dict[str, Any]] = []

    for _, row in selection_df.iterrows():
        short_name = str(row.get("shortName", "")).strip()
        if not short_name:
            raise ValueError("Encountered row without shortName in selection input.")

        city = str(row.get("city", "")).strip()
        year_raw = row.get("year", "")
        year_val = ""
        if pd.notna(year_raw) and str(year_raw).strip() != "":
            try:
                year_val = int(float(year_raw))
            except Exception:
                year_val = str(year_raw)

        image_path_raw = str(row.get("image_path", "")).strip()
        if not image_path_raw:
            raise ValueError(f"Selection row {short_name} has empty image_path.")
        image_path = _resolve_image_path(image_path_raw, run_dir=run_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file for {short_name} not found: {image_path}"
            )

        source_crs, source_transform, source_width, source_height = (
            _load_source_georeference(source_image_path=image_path)
        )

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        width, height = image.size
        if width != source_width or height != source_height:
            raise ValueError(
                "Source dimensions mismatch between PIL and raster reader for "
                f"{image_path}: PIL={width}x{height}, raster={source_width}x{source_height}"
            )
        for patch_idx, primary_anchor in enumerate(primary_anchors, start=1):
            patch_id = f"{short_name}_p{patch_idx}"
            candidate_anchors = [primary_anchor] + fallback_anchor_list

            final_img: Image.Image | None = None
            final_window: PatchWindow | None = None
            final_anchor: tuple[float, float] | None = None
            final_qc: PatchQCDecision | None = None
            replacement_reason = ""
            fallback_used = False
            reject_reasons: list[str] = []

            for candidate_idx, anchor in enumerate(candidate_anchors):
                try:
                    window = _compute_patch_window(
                        width=width,
                        height=height,
                        patch_size=int(patch_size),
                        anchor=anchor,
                    )
                except ValueError:
                    reject_reasons.append("tile_smaller_than_patch")
                    continue

                patch = image.crop((window.x0, window.y0, window.x1, window.y1))
                qc = _evaluate_patch_qc(patch, qc_mode=qc_mode)

                if qc.passed:
                    final_img = patch
                    final_window = window
                    final_anchor = anchor
                    final_qc = qc
                    fallback_used = candidate_idx > 0
                    if fallback_used and reject_reasons:
                        replacement_reason = reject_reasons[0]
                    break

                reject_reasons.append(str(qc.reason or "qc_failed"))
                final_img = patch
                final_window = window
                final_anchor = anchor
                final_qc = qc

            if (
                final_img is None
                or final_window is None
                or final_anchor is None
                or final_qc is None
            ):
                raise RuntimeError(f"Failed to build patch candidates for {patch_id}")

            qc_status = "qc_passed" if final_qc.passed else "qc_rejected"
            qc_reason = "" if final_qc.passed else str(final_qc.reason or "qc_failed")
            if not final_qc.passed and reject_reasons:
                qc_reason = ";".join(reject_reasons)

            quicklook_rel = Path("quicklooks") / f"{patch_id}.tif"
            quicklook_abs = out_dir / quicklook_rel
            _write_patch_quicklook_geotiff(
                patch_img=final_img,
                source_crs=source_crs,
                source_transform=source_transform,
                window=final_window,
                output_path=quicklook_abs,
            )

            record = {
                "patch_id": patch_id,
                "tile_shortname": short_name,
                "tile_city": city,
                "tile_year": year_val,
                "selection_rank": int(float(row.get("selection_rank", patch_idx))),
                "selection_group": str(row.get("selection_group", "core")),
                "patch_index": int(patch_idx),
                "patch_size_px": int(patch_size),
                "image_path": image_path_raw,
                "resolved_image_path": str(image_path.resolve()),
                "tile_width_px": int(width),
                "tile_height_px": int(height),
                "primary_anchor_x": float(primary_anchor[0]),
                "primary_anchor_y": float(primary_anchor[1]),
                "selected_anchor_x": float(final_anchor[0]),
                "selected_anchor_y": float(final_anchor[1]),
                "fallback_used": bool(fallback_used),
                "replacement_reason": replacement_reason,
                "x0": int(final_window.x0),
                "y0": int(final_window.y0),
                "x1": int(final_window.x1),
                "y1": int(final_window.y1),
                "qc_status": qc_status,
                "qc_reason": qc_reason,
                "quicklook_path": str(quicklook_rel),
                "split_fold": pd.NA,
                "qc_std_gray": float(final_qc.metrics.get("std_gray", float("nan"))),
                "qc_dynamic_range": float(
                    final_qc.metrics.get("dynamic_range", float("nan"))
                ),
                "qc_edge_dark": float(final_qc.metrics.get("edge_dark", float("nan"))),
                "qc_center_dark": float(
                    final_qc.metrics.get("center_dark", float("nan"))
                ),
            }
            records.append(record)

    manifest_df = pd.DataFrame(records)
    manifest_df = manifest_df.sort_values(
        ["selection_rank", "patch_index", "patch_id"], kind="stable"
    ).reset_index(drop=True)

    split_manifest_path = out_dir / "patch_split_manifest.json"
    manifest_df, split_manifest_sha = _build_split_manifest(
        manifest_df=manifest_df,
        run_id=run_path.name,
        output_path=split_manifest_path,
        n_splits=split_n_splits,
    )

    manifest_csv_path = out_dir / "patch_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)

    manifest_json_path = out_dir / "patch_manifest.json"
    _write_patch_manifest_json(
        out_path=manifest_json_path,
        records=manifest_df.to_dict(orient="records"),
        run_id=run_path.name,
        patch_size=patch_size,
        patches_per_tile=patches_per_tile,
        include_case=include_case,
        qc_mode=qc_mode,
        fallback_anchors=fallback_anchor_list,
    )

    qc_report_path = out_dir / "patch_qc_report.csv"
    manifest_df[
        [
            "patch_id",
            "tile_shortname",
            "selection_group",
            "patch_index",
            "qc_status",
            "qc_reason",
            "fallback_used",
            "replacement_reason",
            "selected_anchor_x",
            "selected_anchor_y",
            "quicklook_path",
            "split_fold",
        ]
    ].to_csv(qc_report_path, index=False)

    scaffolding_paths = _write_annotation_scaffolding(
        output_dir=out_dir,
        manifest_df=manifest_df,
    )

    contract_payload = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "dataset_version": dataset_version,
        "patch_policy_version": patch_policy_version,
        "settings": {
            "patch_size_px": int(patch_size),
            "patches_per_tile": int(patches_per_tile),
            "include_case": bool(include_case),
            "fallback_anchors": [list(anchor) for anchor in fallback_anchor_list],
            "qc_mode": qc_mode,
            "splitter": "GroupKFold",
            "split_n_splits": int(split_n_splits),
        },
        "source_files": {
            key: _rel_to(run_path, path) for key, path in source_files.items()
        },
        "source_hashes": source_hashes,
        "artifacts": {
            "patch_manifest_csv": _rel_to(run_path, manifest_csv_path),
            "patch_manifest_json": _rel_to(run_path, manifest_json_path),
            "patch_qc_report_csv": _rel_to(run_path, qc_report_path),
            "patch_split_manifest_json": _rel_to(run_path, split_manifest_path),
            "class_mapping_yaml": _rel_to(run_path, scaffolding_paths["class_mapping"]),
            "annotation_progress_csv": _rel_to(
                run_path, scaffolding_paths["annotation_progress"]
            ),
            "annotation_qa_log_csv": _rel_to(
                run_path, scaffolding_paths["annotation_qa_log"]
            ),
        },
        "counts": {
            "tiles_total": int(manifest_df["tile_shortname"].nunique()),
            "patches_total": int(len(manifest_df)),
            "patches_qc_passed": int((manifest_df["qc_status"] == "qc_passed").sum()),
            "patches_qc_rejected": int((manifest_df["qc_status"] != "qc_passed").sum()),
            "quicklook_geotiff_count": int(len(manifest_df)),
        },
        "split_manifest_sha256": split_manifest_sha,
    }

    contract_path = out_dir / "annotation_dataset_contract.json"
    contract_path.write_text(
        json.dumps(contract_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_path),
        "output_dir": str(out_dir),
        "patch_manifest_csv": str(manifest_csv_path),
        "patch_manifest_json": str(manifest_json_path),
        "patch_qc_report_csv": str(qc_report_path),
        "patch_split_manifest_json": str(split_manifest_path),
        "annotation_dataset_contract_json": str(contract_path),
        "patches_total": int(len(manifest_df)),
        "patches_qc_passed": int((manifest_df["qc_status"] == "qc_passed").sum()),
        "patches_qc_rejected": int((manifest_df["qc_status"] != "qc_passed").sum()),
        "split_manifest_sha256": split_manifest_sha,
    }


@cli_command(
    "thesis-build-annotation-plan",
    help="Build deterministic patch manifest + split contract for thesis annotation.",
    args={
        "run_dir": {
            "type": str,
            "required": True,
            "help": "Path to canonical thesis run directory (contains selection_*.csv)",
        },
        "patch_size": {
            "type": int,
            "default": 1024,
            "help": "Square patch size in pixels",
        },
        "patches_per_tile": {
            "type": int,
            "default": 2,
            "help": "Number of patches to sample per tile",
        },
        "include_case": {
            "type": str,
            "default": "true",
            "choices": ["true", "false"],
            "help": "Include case tiles in manifest (true|false)",
        },
        "fallback_anchors": {
            "type": str,
            "default": "0.25,0.75;0.75,0.25;0.50,0.25;0.50,0.75",
            "help": "Fallback anchors as 'x,y;x,y' normalized coordinates",
        },
        "qc_mode": {
            "type": str,
            "default": "heuristic_v1",
            "choices": ["heuristic_v1", "none"],
            "help": "Patch quality check mode",
        },
        "output_subdir": {
            "type": str,
            "default": "annotation_plan",
            "help": "Output subdirectory inside run_dir",
        },
        "split_n_splits": {
            "type": int,
            "default": 5,
            "help": "Number of GroupKFold splits",
        },
        "dataset_version": {
            "type": str,
            "default": "annotation_plan_v1",
            "help": "Dataset version label for contract metadata",
        },
        "patch_policy_version": {
            "type": str,
            "default": "kdr_patch_policy_v1",
            "help": "Patch policy version label for contract metadata",
        },
    },
)
def main(
    run_dir: str,
    patch_size: int = 1024,
    patches_per_tile: int = 2,
    include_case: str = "true",
    fallback_anchors: str = "0.25,0.75;0.75,0.25;0.50,0.25;0.50,0.75",
    qc_mode: str = "heuristic_v1",
    output_subdir: str = "annotation_plan",
    split_n_splits: int = 5,
    dataset_version: str = "annotation_plan_v1",
    patch_policy_version: str = "kdr_patch_policy_v1",
) -> int:
    include_case_bool = _parse_bool_flag(include_case)

    summary = run_thesis_build_annotation_plan(
        run_dir=run_dir,
        patch_size=patch_size,
        patches_per_tile=patches_per_tile,
        include_case=include_case_bool,
        fallback_anchors=fallback_anchors,
        qc_mode=qc_mode,
        output_subdir=output_subdir,
        split_n_splits=split_n_splits,
        dataset_version=dataset_version,
        patch_policy_version=patch_policy_version,
    )

    print("✅ Annotation plan generated")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0
