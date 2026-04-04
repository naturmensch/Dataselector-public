from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

WORKFLOW_VERSION = "phase5_width_calibration_v2"
SUPPORTED_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 9)
COMMON_CLASS_QUOTAS: dict[int, int] = {0: 12, 1: 12, 6: 12}
MID_CLASS_QUOTAS: dict[int, int] = {2: 9, 5: 9}
SPECIAL_CLASS_QUOTAS: dict[int, int] = {9: 9}
REPEAT_QUOTAS: dict[int, int] = {0: 2, 1: 2, 6: 2, 2: 1, 5: 1, 9: 1}
REPEAT_ALL_CLASSES: set[int] = {3, 4, 8}
AUDIT_RARE_CLASSES: set[int] = {3, 4, 8, 9}
HAMBURG_TILE = "KDR_146"
DEFAULT_DISPLAY_CROP_FACTOR = 1.0
DEFAULT_DISPLAY_SCALE = 4
TASK_FILENAME = "width_calibration_tasks.csv"
MANIFEST_FILENAME = "width_calibration_manifest.json"
MEASUREMENTS_FILENAME = "width_calibration_measurements.csv"
SUMMARY_FILENAME = "width_calibration_summary.csv"
SUMMARY_JSON_FILENAME = "width_calibration_summary.json"
SENSITIVITY_FILENAME = "width_calibration_sensitivity.csv"
SENSITIVITY_OVERLAY_DIRNAME = "width_calibration_sensitivity_overlays"
ARCHIVE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TASK_COLUMNS = [
    "task_id",
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
    "queue_position",
    "pass_type",
    "repeat_of_task_id",
]
MEASUREMENT_COLUMNS = [
    "task_id",
    "candidate_id",
    "class",
    "patch_id",
    "tile_shortname",
    "source_fid",
    "source_feature_id",
    "measure_id",
    "pass_type",
    "repeat_of_task_id",
    "anchor_x_px",
    "anchor_y_px",
    "click1_x_px",
    "click1_y_px",
    "click2_x_px",
    "click2_y_px",
    "width_px",
    "keep",
    "reject_reason",
    "note",
]
SUMMARY_COLUMNS = [
    "class",
    "n_valid_primary",
    "median_px",
    "IQR_px",
    "MAD_px",
    "repeat_median_abs_diff_px",
    "low_evidence_flag",
    "high_variance_flag",
    "low_reliability_flag",
    "final_width_px",
]
SENSITIVITY_COLUMNS = [
    "patch_id",
    "audit_reason",
    "variant",
    "classes_present",
    "foreground_pixels",
    "connected_components",
    "delta_foreground_pixels_vs_baseline",
    "delta_connected_components_vs_baseline",
]
REJECTION_REASONS = [
    "crossing",
    "label_overlap",
    "endpoint",
    "tight_curve",
    "blur_damage",
    "ambiguous_symbol",
    "crop_too_small",
    "click_error",
    "other",
]
REQUIRED_PATCH_COLUMNS = [
    "patch_id",
    "tile_shortname",
    "selection_rank",
    "patch_index",
    "quicklook_path",
]
LEGACY_TASK_REQUIRED_COLUMNS = [col for col in TASK_COLUMNS if col != "source_fid"]
LEGACY_MEASUREMENT_REQUIRED_COLUMNS = [
    col for col in MEASUREMENT_COLUMNS if col != "source_fid"
]
VIEWER_HOTKEY_HELP = (
    "Hotkeys: left-click x2 measure | r reject | s skip | "
    "u undo | q quit | Esc clear"
)


@dataclass(frozen=True)
class EligibilityParameters:
    endpoint_exclusion_fraction: float = 0.20
    minimum_border_margin_factor: float = 0.50
    minimum_in_crop_line_support_px: float = 32.0
    anchor_positions: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)

    def border_margin_px(self, crop_size_px: int) -> int:
        return int(np.ceil(float(crop_size_px) * self.minimum_border_margin_factor))


@dataclass(frozen=True)
class PatchContext:
    patch_id: str
    tile_shortname: str
    quicklook_relpath: str
    quicklook_path: Path
    width: int
    height: int
    transform: Any
    crs_wkt: str
    pixel_size: float
    bounds: tuple[float, float, float, float]
    selection_rank: int
    patch_index: int
    patch_order: int


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    candidate_id: str
    class_id: int
    patch_id: str
    tile_shortname: str
    source_fid: str
    source_feature_id: str
    quicklook_path: str
    anchor_x_px: int
    anchor_y_px: int
    crop_size_px: int
    queue_position: int
    pass_type: str
    repeat_of_task_id: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> TaskRecord:
        return cls(
            task_id=str(row["task_id"]),
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
            queue_position=int(row["queue_position"]),
            pass_type=str(row["pass_type"]),
            repeat_of_task_id=str(row["repeat_of_task_id"]),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "class": self.class_id,
            "patch_id": self.patch_id,
            "tile_shortname": self.tile_shortname,
            "source_fid": self.source_fid,
            "source_feature_id": self.source_feature_id,
            "quicklook_path": self.quicklook_path,
            "anchor_x_px": self.anchor_x_px,
            "anchor_y_px": self.anchor_y_px,
            "crop_size_px": self.crop_size_px,
            "queue_position": self.queue_position,
            "pass_type": self.pass_type,
            "repeat_of_task_id": self.repeat_of_task_id,
        }


@dataclass(frozen=True)
class MeasurementRecord:
    task_id: str
    candidate_id: str
    class_id: int
    patch_id: str
    tile_shortname: str
    source_fid: str
    source_feature_id: str
    measure_id: str
    pass_type: str
    repeat_of_task_id: str
    anchor_x_px: int
    anchor_y_px: int
    click1_x_px: Any
    click1_y_px: Any
    click2_x_px: Any
    click2_y_px: Any
    width_px: Any
    keep: int
    reject_reason: str
    note: str

    def to_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "class": self.class_id,
            "patch_id": self.patch_id,
            "tile_shortname": self.tile_shortname,
            "source_fid": self.source_fid,
            "source_feature_id": self.source_feature_id,
            "measure_id": self.measure_id,
            "pass_type": self.pass_type,
            "repeat_of_task_id": self.repeat_of_task_id,
            "anchor_x_px": self.anchor_x_px,
            "anchor_y_px": self.anchor_y_px,
            "click1_x_px": self.click1_x_px,
            "click1_y_px": self.click1_y_px,
            "click2_x_px": self.click2_x_px,
            "click2_y_px": self.click2_y_px,
            "width_px": self.width_px,
            "keep": self.keep,
            "reject_reason": self.reject_reason,
            "note": self.note,
        }


@dataclass(frozen=True)
class SummaryRow:
    class_id: int
    n_valid_primary: int
    median_px: Any
    IQR_px: Any
    MAD_px: Any
    repeat_median_abs_diff_px: Any
    low_evidence_flag: bool
    high_variance_flag: bool
    low_reliability_flag: bool
    final_width_px: Any

    def to_row(self) -> dict[str, Any]:
        return {
            "class": self.class_id,
            "n_valid_primary": self.n_valid_primary,
            "median_px": self.median_px,
            "IQR_px": self.IQR_px,
            "MAD_px": self.MAD_px,
            "repeat_median_abs_diff_px": self.repeat_median_abs_diff_px,
            "low_evidence_flag": self.low_evidence_flag,
            "high_variance_flag": self.high_variance_flag,
            "low_reliability_flag": self.low_reliability_flag,
            "final_width_px": self.final_width_px,
        }


@dataclass(frozen=True)
class SensitivityRow:
    patch_id: str
    audit_reason: str
    variant: str
    classes_present: str
    foreground_pixels: int
    connected_components: int
    delta_foreground_pixels_vs_baseline: int
    delta_connected_components_vs_baseline: int

    def to_row(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "audit_reason": self.audit_reason,
            "variant": self.variant,
            "classes_present": self.classes_present,
            "foreground_pixels": self.foreground_pixels,
            "connected_components": self.connected_components,
            "delta_foreground_pixels_vs_baseline": self.delta_foreground_pixels_vs_baseline,
            "delta_connected_components_vs_baseline": self.delta_connected_components_vs_baseline,
        }


@dataclass(frozen=True)
class SyncMetadata:
    source_gpkg_path: str
    dest_gpkg_path: str
    roads_layer: str
    source_gpkg_sha256: str
    dest_gpkg_sha256: str
    source_mtime_utc: str
    synced_at_utc: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SyncMetadata:
        return cls(
            source_gpkg_path=str(payload.get("source_gpkg_path", "")),
            dest_gpkg_path=str(payload.get("dest_gpkg_path", "")),
            roads_layer=str(payload.get("roads_layer", "")),
            source_gpkg_sha256=str(payload.get("source_gpkg_sha256", "")),
            dest_gpkg_sha256=str(payload.get("dest_gpkg_sha256", "")),
            source_mtime_utc=str(payload.get("source_mtime_utc", "")),
            synced_at_utc=str(payload.get("synced_at_utc", "")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_gpkg_path": self.source_gpkg_path,
            "dest_gpkg_path": self.dest_gpkg_path,
            "roads_layer": self.roads_layer,
            "source_gpkg_sha256": self.source_gpkg_sha256,
            "dest_gpkg_sha256": self.dest_gpkg_sha256,
            "source_mtime_utc": self.source_mtime_utc,
            "synced_at_utc": self.synced_at_utc,
        }


@dataclass(frozen=True)
class RunManifest:
    workflow_version: str
    generated_utc: str
    handoff_dir: str
    roads_gpkg: str
    roads_gpkg_sha256: str
    roads_layer: str
    seed: int
    crop_size_px: int
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RunManifest:
        known = {
            "workflow_version",
            "generated_utc",
            "handoff_dir",
            "roads_gpkg",
            "roads_gpkg_sha256",
            "roads_layer",
            "seed",
            "crop_size_px",
        }
        extras = {str(k): v for k, v in payload.items() if str(k) not in known}
        return cls(
            workflow_version=str(payload.get("workflow_version", "")),
            generated_utc=str(payload.get("generated_utc", "")),
            handoff_dir=str(payload.get("handoff_dir", "")),
            roads_gpkg=str(payload.get("roads_gpkg", "")),
            roads_gpkg_sha256=str(payload.get("roads_gpkg_sha256", "")),
            roads_layer=str(payload.get("roads_layer", "")),
            seed=int(payload.get("seed", 0)),
            crop_size_px=int(payload.get("crop_size_px", 0)),
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "workflow_version": self.workflow_version,
            "generated_utc": self.generated_utc,
            "handoff_dir": self.handoff_dir,
            "roads_gpkg": self.roads_gpkg,
            "roads_gpkg_sha256": self.roads_gpkg_sha256,
            "roads_layer": self.roads_layer,
            "seed": self.seed,
            "crop_size_px": self.crop_size_px,
        }
        payload.update(self.extras)
        return payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def resolve_path(
    path_value: str | Path,
    *,
    repo_root_path: Path,
    prefer_repo: bool = False,
) -> Path:
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate
    in_repo = repo_root_path / candidate
    if prefer_repo or in_repo.exists():
        return in_repo
    return candidate


def normalize_class(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def normalize_source_fid(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def is_hamburg_patch(patch_id: str, tile_shortname: str) -> bool:
    return str(patch_id).startswith(f"{HAMBURG_TILE}_") or str(tile_shortname) == HAMBURG_TILE


__all__ = [
    "ARCHIVE_TIMESTAMP_FORMAT",
    "AUDIT_RARE_CLASSES",
    "COMMON_CLASS_QUOTAS",
    "DEFAULT_DISPLAY_CROP_FACTOR",
    "DEFAULT_DISPLAY_SCALE",
    "EligibilityParameters",
    "HAMBURG_TILE",
    "LEGACY_MEASUREMENT_REQUIRED_COLUMNS",
    "LEGACY_TASK_REQUIRED_COLUMNS",
    "MANIFEST_FILENAME",
    "MEASUREMENTS_FILENAME",
    "MEASUREMENT_COLUMNS",
    "MID_CLASS_QUOTAS",
    "PatchContext",
    "REJECTION_REASONS",
    "REPEAT_ALL_CLASSES",
    "REPEAT_QUOTAS",
    "REQUIRED_PATCH_COLUMNS",
    "RunManifest",
    "SPECIAL_CLASS_QUOTAS",
    "SUMMARY_COLUMNS",
    "SUMMARY_FILENAME",
    "SUMMARY_JSON_FILENAME",
    "SUPPORTED_CLASSES",
    "SENSITIVITY_COLUMNS",
    "SENSITIVITY_FILENAME",
    "SENSITIVITY_OVERLAY_DIRNAME",
    "SummaryRow",
    "SyncMetadata",
    "TASK_COLUMNS",
    "TASK_FILENAME",
    "TaskRecord",
    "MeasurementRecord",
    "SensitivityRow",
    "VIEWER_HOTKEY_HELP",
    "WORKFLOW_VERSION",
    "is_hamburg_patch",
    "normalize_class",
    "normalize_source_fid",
    "repo_root",
    "require_columns",
    "resolve_path",
    "utc_now",
    "write_json",
]
