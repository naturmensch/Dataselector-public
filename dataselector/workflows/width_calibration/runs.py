from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd

from dataselector.runtime.parameter_snapshot import compute_file_sha256

from .models import (
    ARCHIVE_TIMESTAMP_FORMAT,
    MANIFEST_FILENAME,
    MEASUREMENTS_FILENAME,
    SENSITIVITY_FILENAME,
    SENSITIVITY_OVERLAY_DIRNAME,
    SUMMARY_FILENAME,
    SUMMARY_JSON_FILENAME,
    SyncMetadata,
    RunManifest,
    TASK_FILENAME,
    WORKFLOW_VERSION,
    repo_root,
    resolve_path,
    utc_now,
    write_json,
)


def default_roads_gpkg_path(*, repo_root_path: Path | None = None) -> Path:
    root = repo_root_path if repo_root_path is not None else repo_root()
    return (root / "handoff" / "local_sources" / "cut_fixed_geometry_roads.gpkg").resolve()


def sync_metadata_path(dest_gpkg: Path) -> Path:
    return dest_gpkg.with_suffix(".sync.json")


def read_sync_metadata(path: Path) -> SyncMetadata | None:
    if not path.exists():
        return None
    return SyncMetadata.from_payload(json.loads(path.read_text(encoding="utf-8")))


def read_roads_layer_info(roads_gpkg: Path, *, roads_layer: str) -> dict[str, Any]:
    import pyogrio

    return dict(pyogrio.read_info(roads_gpkg, layer=roads_layer))


def validate_sync_source(roads_gpkg: Path, *, roads_layer: str) -> dict[str, Any]:
    if not roads_gpkg.exists():
        raise FileNotFoundError(f"Road source not found: {roads_gpkg}")
    info = read_roads_layer_info(roads_gpkg, roads_layer=roads_layer)
    fields = {str(value) for value in info.get("fields", [])}
    fid_column = str(info.get("fid_column") or "").strip()
    if "class" not in fields:
        raise ValueError(
            f"Road layer missing required 'class' field: {roads_gpkg} [{roads_layer}]"
        )
    if not fid_column:
        raise ValueError(
            f"Road layer missing required GeoPackage fid column: {roads_gpkg} [{roads_layer}]"
        )
    return info


def resolve_roads_layer_name(roads_gpkg: Path) -> str:
    try:
        layers = gpd.list_layers(roads_gpkg)
    except Exception:
        return "cut_fixed_geometry_roads"
    if len(layers) == 1:
        return str(layers.iloc[0]["name"])
    names = {str(value) for value in layers["name"].tolist()}
    if "cut_fixed_geometry_roads" in names:
        return "cut_fixed_geometry_roads"
    raise ValueError(
        "roads_gpkg contains multiple layers and no default "
        "'cut_fixed_geometry_roads' layer could be resolved."
    )


def source_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def copy_file_atomic(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{dest.stem}.",
        suffix=dest.suffix,
        dir=str(dest.parent),
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, dest)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def build_sync_metadata(
    *,
    source_gpkg: Path,
    dest_gpkg: Path,
    roads_layer: str,
    source_sha256: str,
    dest_sha256: str,
) -> SyncMetadata:
    return SyncMetadata(
        source_gpkg_path=str(source_gpkg),
        dest_gpkg_path=str(dest_gpkg),
        roads_layer=str(roads_layer),
        source_gpkg_sha256=str(source_sha256),
        dest_gpkg_sha256=str(dest_sha256),
        source_mtime_utc=source_mtime_utc(source_gpkg),
        synced_at_utc=utc_now(),
    )


def run_archive_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(ARCHIVE_TIMESTAMP_FORMAT)


def list_width_calibration_artifacts(out_dir: Path) -> list[Path]:
    if not out_dir.exists() or not out_dir.is_dir():
        return []
    artifacts: list[Path] = []
    for path in out_dir.iterdir():
        name = path.name
        if name in {
            TASK_FILENAME,
            MANIFEST_FILENAME,
            MEASUREMENTS_FILENAME,
            SUMMARY_FILENAME,
            SUMMARY_JSON_FILENAME,
            SENSITIVITY_FILENAME,
            SENSITIVITY_OVERLAY_DIRNAME,
        } or name.startswith("width_calibration_"):
            artifacts.append(path)
    return sorted(artifacts)


def archive_run_dir_path(out_dir: Path) -> Path:
    base = out_dir.parent / f"{out_dir.name}_archive_{run_archive_timestamp()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = out_dir.parent / f"{base.name}_{suffix:02d}"
        suffix += 1
    return candidate


def load_width_calibration_manifest(tasks_csv_path: Path) -> tuple[Path, RunManifest]:
    manifest_path = tasks_csv_path.parent / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing width calibration manifest next to tasks CSV: {manifest_path}"
        )
    return manifest_path, RunManifest.from_payload(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )


def resolve_manifest_path(path_value: str | Path, *, repo_root_path: Path) -> Path:
    return resolve_path(path_value, repo_root_path=repo_root_path, prefer_repo=False).resolve()


def validated_sync_metadata_for_local_copy(
    roads_gpkg: Path,
    *,
    local_sha256: str,
) -> tuple[Path | None, SyncMetadata | None]:
    sync_path = sync_metadata_path(roads_gpkg)
    sync_metadata = read_sync_metadata(sync_path)
    if sync_metadata is None:
        return None, None
    recorded_dest = str(sync_metadata.dest_gpkg_path).strip()
    if recorded_dest:
        recorded_dest_path = Path(recorded_dest)
        if not recorded_dest_path.is_absolute():
            recorded_dest_path = resolve_path(
                recorded_dest_path,
                repo_root_path=repo_root(),
                prefer_repo=False,
            ).resolve()
        if recorded_dest_path != roads_gpkg.resolve():
            return None, None
    if sync_metadata.dest_gpkg_sha256 and sync_metadata.dest_gpkg_sha256 != local_sha256:
        return None, None
    return sync_path, sync_metadata


def prepare_sync_metadata_for_manifest(
    roads_gpkg_path: Path,
    *,
    roads_gpkg_sha256: str,
) -> tuple[Path | None, SyncMetadata | None]:
    return validated_sync_metadata_for_local_copy(
        roads_gpkg_path,
        local_sha256=roads_gpkg_sha256,
    )


def prompt_yes_no(message: str, *, default: bool = False) -> bool:
    if not sys.stdin.isatty():
        return default
    prompt = "[Y/n]" if default else "[y/N]"
    while True:
        reply = input(f"{message} {prompt} ").strip().lower()
        if not reply:
            return default
        if reply in {"y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def sync_width_calibration_source(
    *,
    source_gpkg: str | Path,
    dest_gpkg: str | Path | None = None,
    roads_layer: str = "cut_fixed_geometry_roads",
) -> dict[str, Any]:
    repo_root_path = repo_root()
    source_path = resolve_path(source_gpkg, repo_root_path=repo_root_path, prefer_repo=False).resolve()
    dest_path = (
        resolve_path(dest_gpkg, repo_root_path=repo_root_path, prefer_repo=True).resolve()
        if dest_gpkg is not None
        else default_roads_gpkg_path(repo_root_path=repo_root_path)
    )
    validate_sync_source(source_path, roads_layer=roads_layer)
    source_sha256 = compute_file_sha256(source_path)
    dest_exists = dest_path.exists()
    dest_sha256_before = compute_file_sha256(dest_path) if dest_exists else ""
    in_sync = bool(dest_exists and dest_sha256_before == source_sha256)
    copied = False
    if not in_sync:
        copy_file_atomic(source_path, dest_path)
        copied = True
    dest_sha256_after = compute_file_sha256(dest_path)
    metadata = build_sync_metadata(
        source_gpkg=source_path,
        dest_gpkg=dest_path,
        roads_layer=roads_layer,
        source_sha256=source_sha256,
        dest_sha256=dest_sha256_after,
    )
    sync_path = sync_metadata_path(dest_path)
    write_json(sync_path, metadata.to_payload())
    return {
        "source_gpkg": str(source_path),
        "dest_gpkg": str(dest_path),
        "roads_layer": str(roads_layer),
        "source_gpkg_sha256": source_sha256,
        "dest_gpkg_sha256": dest_sha256_after,
        "sync_metadata_path": str(sync_path),
        "in_sync": bool(in_sync),
        "copied": bool(copied),
    }


def maybe_sync_local_copy_from_source(
    roads_gpkg_path: Path,
    *,
    roads_layer: str,
    current_local_sha: str,
    prompt_for_sync: bool,
) -> tuple[str, bool]:
    sync_path, sync_metadata = validated_sync_metadata_for_local_copy(
        roads_gpkg_path,
        local_sha256=current_local_sha,
    )
    if sync_path is None or sync_metadata is None:
        return current_local_sha, False
    source_value = str(sync_metadata.source_gpkg_path).strip()
    if not source_value:
        return current_local_sha, False
    repo_root_path = repo_root()
    source_gpkg_path = resolve_manifest_path(source_value, repo_root_path=repo_root_path)
    if not source_gpkg_path.exists():
        return current_local_sha, False
    current_source_sha = compute_file_sha256(source_gpkg_path)
    if current_source_sha == current_local_sha:
        return current_local_sha, False
    if not prompt_for_sync:
        raise RuntimeError(
            "The repo-local roads copy is stale relative to the current source GeoPackage. "
            "Run sync-width-calibration-source first, then rerun prepare-width-calibration."
        )
    allow_sync = prompt_yes_no(
        f"Detected source changes in {source_gpkg_path}. Sync the repo-local roads copy now?",
        default=False,
    )
    if not allow_sync:
        raise RuntimeError(
            "Aborted because the repo-local roads copy is stale relative to the current source "
            "GeoPackage. Run sync-width-calibration-source first, then rerun prepare-width-calibration."
        )
    sync_width_calibration_source(
        source_gpkg=source_gpkg_path,
        dest_gpkg=roads_gpkg_path,
        roads_layer=roads_layer,
    )
    return compute_file_sha256(roads_gpkg_path), True


def existing_run_manifest_staleness_reason(
    out_dir: Path,
    *,
    handoff_dir: Path,
    roads_layer: str,
    seed: int,
    crop_size_px: int,
    current_local_sha: str,
    current_sync_metadata: SyncMetadata | None,
) -> str | None:
    artifacts = list_width_calibration_artifacts(out_dir)
    if not artifacts:
        return None
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return "existing width-calibration artifacts were found, but the run manifest is missing"
    try:
        manifest = RunManifest.from_payload(json.loads(manifest_path.read_text(encoding="utf-8")))
    except Exception:
        return "existing width-calibration artifacts were found, but the run manifest is unreadable"
    if manifest.workflow_version != WORKFLOW_VERSION:
        return "workflow_version changed"
    if not manifest.handoff_dir:
        return "handoff_dir is missing from the existing run manifest"
    try:
        recorded_handoff_path = resolve_manifest_path(manifest.handoff_dir, repo_root_path=repo_root())
    except Exception:
        return "handoff_dir in the existing run manifest is invalid"
    if recorded_handoff_path != handoff_dir:
        return "handoff_dir changed"
    if manifest.roads_layer != str(roads_layer):
        return "roads_layer changed"
    if manifest.roads_gpkg_sha256 != current_local_sha:
        return "roads_gpkg_sha256 changed"
    if int(manifest.seed) != int(seed):
        return "seed changed"
    if int(manifest.crop_size_px) != int(crop_size_px):
        return "crop_size_px changed"
    current_source_sha = ""
    if current_sync_metadata is not None:
        current_source_sha = str(current_sync_metadata.source_gpkg_sha256).strip()
        if not current_source_sha:
            return "current sync metadata is incomplete"
    recorded_source_sha = str(manifest.extras.get("sync_source_gpkg_sha256", "")).strip()
    if current_source_sha and recorded_source_sha != current_source_sha:
        return "sync_source_gpkg_sha256 changed"
    return None


def maybe_archive_stale_prepare_run(
    out_dir: Path,
    *,
    handoff_dir: Path,
    roads_layer: str,
    seed: int,
    crop_size_px: int,
    current_local_sha: str,
    current_sync_metadata: SyncMetadata | None,
    prompt_for_archive: bool,
) -> Path | None:
    stale_reason = existing_run_manifest_staleness_reason(
        out_dir,
        handoff_dir=handoff_dir,
        roads_layer=roads_layer,
        seed=seed,
        crop_size_px=crop_size_px,
        current_local_sha=current_local_sha,
        current_sync_metadata=current_sync_metadata,
    )
    if stale_reason is None:
        return None
    archive_path = archive_run_dir_path(out_dir)
    message = (
        f"Archive stale width-calibration run to {archive_path} and rebuild active out_dir? "
        f"Detected stale reason: {stale_reason}"
    )
    if not prompt_for_archive or not sys.stdin.isatty():
        raise RuntimeError(f"{message} Rerun interactively to confirm archival.")
    if not prompt_yes_no(message, default=False):
        raise RuntimeError(
            "Aborted because the existing width-calibration run is stale and archival was declined."
        )
    shutil.move(str(out_dir), str(archive_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    return archive_path


def preflight_measure_width_calibration(
    *,
    tasks_csv_path: Path,
    repo_root_path: Path,
    prompt_for_sync: bool = False,
) -> tuple[RunManifest, Path]:
    _manifest_path, manifest = load_width_calibration_manifest(tasks_csv_path)
    if not manifest.roads_gpkg or not manifest.roads_gpkg_sha256:
        raise RuntimeError(
            "Width calibration manifest is missing roads_gpkg or roads_gpkg_sha256. "
            "Rerun prepare-width-calibration."
        )
    roads_gpkg_path = resolve_manifest_path(manifest.roads_gpkg, repo_root_path=repo_root_path)
    if not roads_gpkg_path.exists():
        raise FileNotFoundError(
            f"Roads GeoPackage recorded in width calibration manifest was not found: {roads_gpkg_path}"
        )
    current_local_sha = compute_file_sha256(roads_gpkg_path)
    current_local_sha, synced_from_source = maybe_sync_local_copy_from_source(
        roads_gpkg_path,
        roads_layer=str(manifest.roads_layer or "cut_fixed_geometry_roads"),
        current_local_sha=current_local_sha,
        prompt_for_sync=prompt_for_sync,
    )
    if synced_from_source and current_local_sha != manifest.roads_gpkg_sha256:
        raise RuntimeError(
            "The repo-local roads copy was synced from the editable source, but the active "
            "width-calibration task queue is now stale. Rerun prepare-width-calibration."
        )
    if current_local_sha != manifest.roads_gpkg_sha256:
        raise RuntimeError(
            "The active width-calibration task queue is stale relative to the current "
            "repo-local roads copy. Rerun prepare-width-calibration."
        )
    sync_path, sync_metadata = validated_sync_metadata_for_local_copy(
        roads_gpkg_path,
        local_sha256=current_local_sha,
    )
    if sync_path is None or sync_metadata is None:
        return manifest, roads_gpkg_path
    source_value = str(sync_metadata.source_gpkg_path).strip()
    if not source_value:
        return manifest, roads_gpkg_path
    source_gpkg_path = resolve_manifest_path(source_value, repo_root_path=repo_root_path)
    if not source_gpkg_path.exists():
        return manifest, roads_gpkg_path
    current_source_sha = compute_file_sha256(source_gpkg_path)
    if current_source_sha != current_local_sha:
        raise RuntimeError(
            "The repo-local roads copy is stale relative to the current source GeoPackage. "
            "Run sync-width-calibration-source first, then rerun prepare-width-calibration."
        )
    return manifest, roads_gpkg_path


__all__ = [
    "archive_run_dir_path",
    "default_roads_gpkg_path",
    "load_width_calibration_manifest",
    "maybe_archive_stale_prepare_run",
    "maybe_sync_local_copy_from_source",
    "prepare_sync_metadata_for_manifest",
    "preflight_measure_width_calibration",
    "resolve_manifest_path",
    "resolve_roads_layer_name",
    "sync_width_calibration_source",
    "sync_metadata_path",
    "validated_sync_metadata_for_local_copy",
]
