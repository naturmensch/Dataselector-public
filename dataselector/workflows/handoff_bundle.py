"""Reusable tile and patch handoff preparation/verification helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXIT_SCHEMA = 2
EXIT_DATA = 3
EXIT_POLICY = 4

DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_TILE_POLICY = "config/tile_exclusion_policy.yaml"

REQUIRED_SELECTED_COLUMNS = [
    "shortName",
    "image_path",
    "image_filename",
    "selection_rank",
]
REQUIRED_MASK_COLUMNS = ["shortName", "required_mask_filename"]
REQUIRED_MANIFEST_FIELDS = [
    "schema_version",
    "selection_id",
    "run_id",
    "run_dir",
    "resolved_snapshot_path",
    "resolved_snapshot_sha256",
    "selection_csv_path",
    "selection_csv_sha256",
    "selection_count",
    "tile_exclusion_policy_path",
    "tile_exclusion_policy_sha256",
    "excluded_tiles",
    "split_authority",
]
REQUIRED_PATCH_SELECTED_COLUMNS = [
    "patch_id",
    "tile_shortname",
    "image_path",
    "image_filename",
    "x0",
    "y0",
    "x1",
    "y1",
    "split_fold",
    "quicklook_path",
]
REQUIRED_PATCH_MASK_COLUMNS = ["patch_id", "required_mask_filename"]
REQUIRED_PATCH_MANIFEST_FIELDS = [
    "schema_version",
    "selection_id",
    "run_id",
    "run_dir",
    "resolved_snapshot_path",
    "resolved_snapshot_sha256",
    "patch_selection_csv_path",
    "patch_selection_csv_sha256",
    "patch_selection_count",
    "patch_mask_requirements_path",
    "patch_mask_requirements_sha256",
    "patch_split_manifest_path",
    "patch_split_manifest_sha256",
    "patch_quicklook_format",
    "tile_exclusion_policy_path",
    "tile_exclusion_policy_sha256",
    "excluded_tiles",
    "split_authority",
]


@dataclass(frozen=True)
class HandoffCheckError(RuntimeError):
    exit_code: int
    messages: tuple[str, ...]

    def __init__(self, exit_code: int, messages: list[str] | tuple[str, ...]):
        object.__setattr__(self, "exit_code", int(exit_code))
        object.__setattr__(self, "messages", tuple(str(msg) for msg in messages))
        super().__init__("\n".join(self.messages))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(
    path_str: str | Path,
    *,
    repo_root: Path,
    run_dir: Path | None = None,
    prefer_repo: bool = False,
) -> Path:
    candidate = Path(str(path_str))
    if candidate.is_absolute():
        return candidate
    if run_dir is not None:
        in_run = run_dir / candidate
        if in_run.exists():
            return in_run
    in_repo = repo_root / candidate
    if prefer_repo or in_repo.exists():
        return in_repo
    return candidate


def _rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _load_excluded_tiles(policy_path: Path) -> list[str]:
    if not policy_path.exists():
        return []
    try:
        import yaml  # type: ignore
    except Exception:
        return []

    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    rules = payload.get("rules", []) if isinstance(payload, dict) else []
    excluded: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", "")).strip().lower()
        if action != "exclude_from_candidate_pool":
            continue
        match = rule.get("match", {})
        if not isinstance(match, dict):
            continue
        values = match.get("shortName", [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            text = str(value).strip()
            if text:
                excluded.append(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in excluded:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_selection_csv(
    *,
    run_dir: Path,
    explicit_selection_csv: str | None,
    repo_root: Path,
) -> tuple[Path, str]:
    explicit = str(explicit_selection_csv or "").strip()
    if explicit:
        path = _resolve_path(
            explicit,
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if not path.exists():
            raise HandoffCheckError(
                EXIT_SCHEMA,
                [f"[SCHEMA] explicit selection CSV not found: {path}"],
            )
        return path, "explicit"

    selection_contract_path = run_dir / "selection_contract.json"
    if selection_contract_path.exists():
        try:
            contract = json.loads(selection_contract_path.read_text(encoding="utf-8"))
            if isinstance(contract, dict):
                core_csv = run_dir / "selection_core.csv"
                if core_csv.exists():
                    return core_csv, "selection_contract_core"
                source_file = str(contract.get("selection_source_file", "")).strip()
                if source_file:
                    source_path = (run_dir / source_file).resolve()
                    if source_path.exists():
                        return source_path, "selection_contract_source_file"
        except Exception:
            pass

    report_path = run_dir / "THESIS_PIPELINE_REPORT.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        import re

        match = re.search(r"Selection file:\s*`([^`]+)`", text)
        if match:
            raw_path = match.group(1).strip()
            if raw_path and raw_path.lower() != "not available":
                candidate = (run_dir / raw_path).resolve()
                if candidate.exists():
                    return candidate, "report"

    tuning_dir = run_dir / "tuning_weights"
    meta_path = tuning_dir / "meta.json"
    if meta_path.exists() and tuning_dir.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            best = meta.get("best_metrics", {}) if isinstance(meta, dict) else {}
            alpha = best.get("alpha")
            beta = best.get("beta")
            gamma = best.get("gamma")
            if alpha is not None and beta is not None and gamma is not None:
                exact_name = f"selection_a{alpha}_b{beta}_g{gamma}.csv"
                exact_path = tuning_dir / exact_name
                if exact_path.exists():
                    return exact_path, "tuning_weights_best_metrics"
                pattern = (
                    f"selection_a{float(alpha):.6f}*_"
                    f"b{float(beta):.6f}*_"
                    f"g{float(gamma):.6f}*.csv"
                )
                candidates = sorted(tuning_dir.glob(pattern))
                if candidates:
                    return candidates[0], "tuning_weights_best_metrics_fuzzy"
        except Exception:
            pass

    fallback = (
        sorted(tuning_dir.glob("selection_a*_b*_g*.csv")) if tuning_dir.exists() else []
    )
    if fallback:
        return fallback[0], "tuning_weights_fallback"

    raise HandoffCheckError(
        EXIT_SCHEMA,
        [
            "[SCHEMA] could not resolve selection CSV (explicit path, report, and tuning fallback failed)"
        ],
    )


def _resolve_patch_manifest_csv(
    *,
    run_dir: Path,
    explicit_path: str | None,
    repo_root: Path,
) -> tuple[Path, str]:
    explicit = str(explicit_path or "").strip()
    if explicit:
        path = _resolve_path(
            explicit,
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if not path.exists():
            raise HandoffCheckError(
                EXIT_SCHEMA,
                [f"[SCHEMA] explicit patch manifest CSV not found: {path}"],
            )
        return path, "explicit"

    default_path = run_dir / "annotation_plan" / "patch_manifest.csv"
    if default_path.exists():
        return default_path, "annotation_plan_default"

    contract_path = run_dir / "annotation_plan" / "annotation_dataset_contract.json"
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            artifact_rel = str(
                (contract.get("artifacts", {}) or {}).get("patch_manifest_csv", "")
            ).strip()
            if artifact_rel:
                candidate = _resolve_path(
                    artifact_rel,
                    repo_root=repo_root,
                    run_dir=run_dir,
                ).resolve()
                if candidate.exists():
                    return candidate, "annotation_plan_contract"
        except Exception:
            pass

    raise HandoffCheckError(
        EXIT_SCHEMA,
        [
            "[SCHEMA] could not resolve patch manifest CSV (explicit path and annotation_plan defaults failed)"
        ],
    )


def _resolve_patch_split_manifest(
    *,
    run_dir: Path,
    explicit_path: str | None,
    repo_root: Path,
) -> tuple[Path, str]:
    explicit = str(explicit_path or "").strip()
    if explicit:
        path = _resolve_path(
            explicit,
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if not path.exists():
            raise HandoffCheckError(
                EXIT_SCHEMA,
                [f"[SCHEMA] explicit patch split manifest not found: {path}"],
            )
        return path, "explicit"

    default_path = run_dir / "annotation_plan" / "patch_split_manifest.json"
    if default_path.exists():
        return default_path, "annotation_plan_default"

    contract_path = run_dir / "annotation_plan" / "annotation_dataset_contract.json"
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            artifact_rel = str(
                (contract.get("artifacts", {}) or {}).get(
                    "patch_split_manifest_json", ""
                )
            ).strip()
            if artifact_rel:
                candidate = _resolve_path(
                    artifact_rel,
                    repo_root=repo_root,
                    run_dir=run_dir,
                ).resolve()
                if candidate.exists():
                    return candidate, "annotation_plan_contract"
        except Exception:
            pass

    raise HandoffCheckError(
        EXIT_SCHEMA,
        [
            "[SCHEMA] could not resolve patch split manifest (explicit path and annotation_plan defaults failed)"
        ],
    )


def _resolve_manifest_artifact(
    *,
    artifact_path: str,
    source_manifest_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> Path:
    artifact = Path(artifact_path)
    if artifact.is_absolute():
        return artifact
    manifest_relative = source_manifest_path.parent / artifact
    if manifest_relative.exists():
        return manifest_relative
    run_relative = run_dir / artifact
    if run_relative.exists():
        return run_relative
    return _resolve_path(artifact_path, repo_root=repo_root, run_dir=run_dir)


def prepare_tile_handoff(
    *,
    run_dir: str | Path,
    out_dir: str | Path,
    selection_csv: str | None = None,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    tile_exclusion_policy: str | Path = DEFAULT_TILE_POLICY,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = (
        Path(repo_root).resolve() if repo_root is not None else _repo_root()
    )
    run_dir_path = _resolve_path(
        run_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    out_dir_path = _resolve_path(
        out_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    tile_policy_path = _resolve_path(
        tile_exclusion_policy,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    _ = _resolve_path(images_dir, repo_root=repo_root_path, prefer_repo=True).resolve()

    if not run_dir_path.exists():
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] run directory not found: {run_dir_path}"],
        )

    out_dir_path.mkdir(parents=True, exist_ok=True)
    source_selection_csv, source_selection = _resolve_selection_csv(
        run_dir=run_dir_path,
        explicit_selection_csv=selection_csv,
        repo_root=repo_root_path,
    )

    try:
        source_rows, _source_fields = _load_csv(source_selection_csv)
    except Exception as exc:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] failed to read selection CSV: {exc}"],
        ) from exc

    if not source_rows:
        raise HandoffCheckError(EXIT_SCHEMA, ["[SCHEMA] selection CSV has no rows"])

    optional_fields = [
        "year",
        "city",
        "city_source",
        "longName",
        "center_x",
        "center_y",
    ]
    normalized_rows: list[dict[str, str | int]] = []
    errors: list[str] = []

    for idx, row in enumerate(source_rows):
        short_name = str(row.get("shortName", "")).strip()
        image_filename = str(
            row.get("image_filename", "") or row.get("filename", "")
        ).strip()
        image_path = str(row.get("image_path", "")).strip()

        if not short_name:
            fallback_name = image_filename or image_path
            if fallback_name:
                short_name = Path(fallback_name).stem
        if not short_name:
            errors.append(
                f"[SCHEMA] row {idx}: missing shortName and no filename fallback"
            )
            continue

        if not image_filename:
            image_filename = (
                Path(image_path).name if image_path else f"{short_name}.png"
            )
        if not image_path:
            image_path = f"data/images/{image_filename}"

        rank_raw = str(row.get("selection_rank", "")).strip()
        try:
            selection_rank = int(float(rank_raw)) if rank_raw else idx
        except Exception:
            selection_rank = idx

        normalized: dict[str, str | int] = {
            "shortName": short_name,
            "image_path": image_path,
            "image_filename": image_filename,
            "selection_rank": selection_rank,
            "_order": idx,
        }
        for field in optional_fields:
            value = str(row.get(field, "")).strip()
            if value:
                normalized[field] = value
        normalized_rows.append(normalized)

    if errors:
        raise HandoffCheckError(EXIT_SCHEMA, errors)

    normalized_rows.sort(
        key=lambda item: (int(item["selection_rank"]), int(item["_order"]))
    )
    seen_short_names: set[str] = set()
    duplicates: list[str] = []
    for row in normalized_rows:
        short_name = str(row["shortName"])
        if short_name in seen_short_names:
            duplicates.append(short_name)
        seen_short_names.add(short_name)
    if duplicates:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [
                f"[SCHEMA] duplicate shortName in selection: {item}"
                for item in sorted(set(duplicates))
            ],
        )

    present_optional = [
        field
        for field in optional_fields
        if any(str(row.get(field, "")).strip() for row in normalized_rows)
    ]
    selected_fieldnames = [
        "shortName",
        "image_path",
        "image_filename",
        "selection_rank",
    ] + present_optional

    selected_maps_path = out_dir_path / "selected_maps.csv"
    with selected_maps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(
                {field: row.get(field, "") for field in selected_fieldnames}
            )

    mask_requirements_path = out_dir_path / "mask_requirements.csv"
    with mask_requirements_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["shortName", "required_mask_filename"]
        )
        writer.writeheader()
        for row in normalized_rows:
            short_name = str(row["shortName"])
            writer.writerow(
                {
                    "shortName": short_name,
                    "required_mask_filename": f"{short_name}_mask.tif",
                }
            )

    selected_maps_sha = _sha256(selected_maps_path)
    source_selection_sha = _sha256(source_selection_csv)

    run_metadata_path = run_dir_path / "run_metadata.json"
    run_metadata: dict[str, Any] = {}
    if run_metadata_path.exists():
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            run_metadata = {}

    extra = run_metadata.get("extra", {}) if isinstance(run_metadata, dict) else {}
    pipeline_snapshot = (
        extra.get("pipeline_metadata_snapshot", {}) if isinstance(extra, dict) else {}
    )
    pipeline_extra = (
        pipeline_snapshot.get("extra", {})
        if isinstance(pipeline_snapshot, dict)
        else {}
    )

    resolved_snapshot_path = str(
        extra.get("resolved_snapshot_path")
        or pipeline_extra.get("resolved_snapshot_path")
        or ""
    ).strip()
    resolved_snapshot_sha = str(
        extra.get("resolved_snapshot_sha256")
        or pipeline_extra.get("resolved_snapshot_sha256")
        or ""
    ).strip()
    if resolved_snapshot_path and not resolved_snapshot_sha:
        resolved_snapshot_candidate = _resolve_path(
            resolved_snapshot_path,
            repo_root=repo_root_path,
            run_dir=run_dir_path,
        )
        if resolved_snapshot_candidate.exists():
            resolved_snapshot_sha = _sha256(resolved_snapshot_candidate)

    excluded_tiles = _load_excluded_tiles(tile_policy_path)
    tile_policy_sha = _sha256(tile_policy_path) if tile_policy_path.exists() else ""

    run_id = run_dir_path.name
    selection_id = f"{run_id}_{selected_maps_sha[:12]}"

    manifest = {
        "schema_version": "handoff_format_v1",
        "selection_id": selection_id,
        "run_id": run_id,
        "run_dir": _rel_or_abs(run_dir_path, repo_root_path),
        "resolved_snapshot_path": resolved_snapshot_path,
        "resolved_snapshot_sha256": resolved_snapshot_sha,
        "selection_csv_path": "selected_maps.csv",
        "selection_csv_sha256": selected_maps_sha,
        "selection_count": len(normalized_rows),
        "tile_exclusion_policy_path": (
            _rel_or_abs(tile_policy_path, repo_root_path)
            if tile_policy_path.exists()
            else ""
        ),
        "tile_exclusion_policy_sha256": tile_policy_sha,
        "excluded_tiles": excluded_tiles,
        "split_authority": "masterarbeit_strassenerkennung_cv",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_selection_csv": _rel_or_abs(source_selection_csv, repo_root_path),
        "source_selection_csv_sha256": source_selection_sha,
        "source_selection_resolution": source_selection,
    }

    handoff_manifest_path = out_dir_path / "handoff_manifest.json"
    handoff_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "run_dir": str(run_dir_path),
        "handoff_dir": str(out_dir_path),
        "selection_source": source_selection,
        "selection_count": len(normalized_rows),
        "selection_id": selection_id,
        "handoff_manifest_path": str(handoff_manifest_path),
        "selected_maps_path": str(selected_maps_path),
        "mask_requirements_path": str(mask_requirements_path),
    }


def prepare_patch_handoff(
    *,
    run_dir: str | Path,
    out_dir: str | Path,
    patch_manifest_csv: str | None = None,
    patch_split_manifest: str | None = None,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    tile_exclusion_policy: str | Path = DEFAULT_TILE_POLICY,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = (
        Path(repo_root).resolve() if repo_root is not None else _repo_root()
    )
    run_dir_path = _resolve_path(
        run_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    out_dir_path = _resolve_path(
        out_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    images_dir_path = _resolve_path(
        images_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    tile_policy_path = _resolve_path(
        tile_exclusion_policy,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()

    if not run_dir_path.exists():
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] run directory not found: {run_dir_path}"],
        )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    source_patch_manifest_csv, source_patch_manifest_resolution = (
        _resolve_patch_manifest_csv(
            run_dir=run_dir_path,
            explicit_path=patch_manifest_csv,
            repo_root=repo_root_path,
        )
    )
    source_patch_split_manifest, source_patch_split_resolution = (
        _resolve_patch_split_manifest(
            run_dir=run_dir_path,
            explicit_path=patch_split_manifest,
            repo_root=repo_root_path,
        )
    )

    try:
        source_rows, source_fields = _load_csv(source_patch_manifest_csv)
    except Exception as exc:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] failed to read patch_manifest.csv: {exc}"],
        ) from exc
    if not source_rows:
        raise HandoffCheckError(
            EXIT_SCHEMA, ["[SCHEMA] patch_manifest.csv has no rows"]
        )

    required_patch_columns = [
        "patch_id",
        "tile_shortname",
        "image_path",
        "x0",
        "y0",
        "x1",
        "y1",
        "qc_status",
        "quicklook_path",
    ]
    missing_columns = [
        col for col in required_patch_columns if col not in source_fields
    ]
    if missing_columns:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [
                f"[SCHEMA] patch_manifest.csv missing column: {col}"
                for col in missing_columns
            ],
        )

    try:
        split_manifest = json.loads(
            source_patch_split_manifest.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] failed to parse patch split manifest: {exc}"],
        ) from exc
    patch_to_fold = split_manifest.get("patch_to_fold", {})
    if not isinstance(patch_to_fold, dict):
        raise HandoffCheckError(
            EXIT_SCHEMA,
            ["[SCHEMA] patch split manifest missing object: patch_to_fold"],
        )

    selected_rows: list[dict[str, str | int]] = []
    errors: list[str] = []
    seen_patch_ids: set[str] = set()

    for idx, row in enumerate(source_rows):
        qc_status = str(row.get("qc_status", "")).strip().lower()
        if qc_status != "qc_passed":
            continue

        patch_id = str(row.get("patch_id", "")).strip()
        tile_shortname = str(row.get("tile_shortname", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        image_filename = str(row.get("image_filename", "")).strip()
        if not image_filename and image_path:
            image_filename = Path(image_path).name

        if not patch_id:
            errors.append(f"[SCHEMA] row {idx}: missing patch_id")
            continue
        if patch_id in seen_patch_ids:
            errors.append(
                f"[SCHEMA] duplicate patch_id in patch_manifest.csv: {patch_id}"
            )
            continue
        seen_patch_ids.add(patch_id)

        if not tile_shortname:
            errors.append(
                f"[SCHEMA] row {idx}: missing tile_shortname for patch_id={patch_id}"
            )
            continue
        if not image_path:
            errors.append(
                f"[SCHEMA] row {idx}: missing image_path for patch_id={patch_id}"
            )
            continue

        quicklook_path = str(row.get("quicklook_path", "")).strip()
        if not quicklook_path:
            errors.append(
                f"[SCHEMA] row {idx}: missing quicklook_path for patch_id={patch_id}"
            )
            continue

        quicklook_rel = Path(quicklook_path)
        if quicklook_rel.is_absolute() or ".." in quicklook_rel.parts:
            errors.append(
                f"[SCHEMA] row {idx}: invalid quicklook_path for patch_id={patch_id}"
            )
            continue
        if quicklook_rel.suffix.lower() != ".tif":
            errors.append(
                f"[SCHEMA] row {idx}: quicklook_path must end with .tif for patch_id={patch_id}"
            )
            continue

        try:
            x0 = int(float(str(row.get("x0", "")).strip()))
            y0 = int(float(str(row.get("y0", "")).strip()))
            x1 = int(float(str(row.get("x1", "")).strip()))
            y1 = int(float(str(row.get("y1", "")).strip()))
        except Exception:
            errors.append(
                f"[SCHEMA] row {idx}: invalid patch bounds for patch_id={patch_id}"
            )
            continue

        split_fold_raw = str(row.get("split_fold", "")).strip()
        if split_fold_raw:
            try:
                split_fold = int(float(split_fold_raw))
            except Exception:
                errors.append(
                    f"[SCHEMA] row {idx}: invalid split_fold for patch_id={patch_id}"
                )
                continue
        else:
            fold_from_manifest = patch_to_fold.get(patch_id)
            if fold_from_manifest is None:
                errors.append(
                    "[SCHEMA] row {}: split_fold missing and not found in patch split manifest for patch_id={}".format(
                        idx, patch_id
                    )
                )
                continue
            try:
                split_fold = int(fold_from_manifest)
            except Exception:
                errors.append(
                    f"[SCHEMA] row {idx}: invalid patch_to_fold mapping for patch_id={patch_id}"
                )
                continue

        patch_index_raw = str(row.get("patch_index", "")).strip()
        try:
            patch_index = int(float(patch_index_raw)) if patch_index_raw else 0
        except Exception:
            patch_index = 0
        selection_rank_raw = str(row.get("selection_rank", "")).strip()
        try:
            selection_rank = (
                int(float(selection_rank_raw)) if selection_rank_raw else idx
            )
        except Exception:
            selection_rank = idx

        selected_rows.append(
            {
                "patch_id": patch_id,
                "tile_shortname": tile_shortname,
                "image_path": image_path,
                "image_filename": image_filename,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "split_fold": split_fold,
                "selection_rank": selection_rank,
                "selection_group": str(row.get("selection_group", "")).strip(),
                "patch_index": patch_index,
                "patch_size_px": str(row.get("patch_size_px", "")).strip(),
                "quicklook_path": str(quicklook_rel),
                "qc_status": str(row.get("qc_status", "")).strip(),
                "qc_reason": str(row.get("qc_reason", "")).strip(),
                "_order": idx,
            }
        )

    if errors:
        raise HandoffCheckError(EXIT_SCHEMA, errors)
    if not selected_rows:
        raise HandoffCheckError(
            EXIT_SCHEMA,
            ["[SCHEMA] patch_manifest.csv has no qc_passed rows"],
        )

    selected_rows.sort(
        key=lambda item: (
            int(item["selection_rank"]),
            int(item["patch_index"]),
            int(item["_order"]),
        )
    )

    quicklook_copy_errors: list[str] = []
    for row in selected_rows:
        patch_id = str(row["patch_id"])
        quicklook_rel = str(row["quicklook_path"]).strip()
        quicklook_src = _resolve_manifest_artifact(
            artifact_path=quicklook_rel,
            source_manifest_path=source_patch_manifest_csv,
            run_dir=run_dir_path,
            repo_root=repo_root_path,
        ).resolve()
        if not quicklook_src.exists():
            quicklook_copy_errors.append(
                f"[SCHEMA] missing quicklook image for patch_id={patch_id}: {quicklook_src}"
            )
            continue
        quicklook_dst = (out_dir_path / quicklook_rel).resolve()
        quicklook_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(quicklook_src, quicklook_dst)

    if quicklook_copy_errors:
        raise HandoffCheckError(EXIT_SCHEMA, quicklook_copy_errors)

    selected_patches_fields = [
        "patch_id",
        "tile_shortname",
        "image_path",
        "image_filename",
        "x0",
        "y0",
        "x1",
        "y1",
        "split_fold",
        "selection_rank",
        "selection_group",
        "patch_index",
        "patch_size_px",
        "quicklook_path",
        "qc_status",
        "qc_reason",
    ]

    selected_patches_path = out_dir_path / "selected_patches.csv"
    with selected_patches_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_patches_fields)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(
                {field: row.get(field, "") for field in selected_patches_fields}
            )

    patch_mask_requirements_path = out_dir_path / "patch_mask_requirements.csv"
    with patch_mask_requirements_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["patch_id", "required_mask_filename"]
        )
        writer.writeheader()
        for row in selected_rows:
            patch_id = str(row["patch_id"])
            writer.writerow(
                {
                    "patch_id": patch_id,
                    "required_mask_filename": f"{patch_id}_mask.tif",
                }
            )

    split_manifest_out_path = out_dir_path / "patch_split_manifest.json"
    split_manifest_out_path.write_text(
        json.dumps(split_manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    selected_patches_sha = _sha256(selected_patches_path)
    patch_mask_requirements_sha = _sha256(patch_mask_requirements_path)
    source_patch_manifest_sha = _sha256(source_patch_manifest_csv)
    source_patch_split_sha = _sha256(source_patch_split_manifest)
    split_manifest_out_sha = _sha256(split_manifest_out_path)

    run_metadata_path = run_dir_path / "run_metadata.json"
    run_metadata: dict[str, Any] = {}
    if run_metadata_path.exists():
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            run_metadata = {}

    extra = run_metadata.get("extra", {}) if isinstance(run_metadata, dict) else {}
    pipeline_snapshot = (
        extra.get("pipeline_metadata_snapshot", {}) if isinstance(extra, dict) else {}
    )
    pipeline_extra = (
        pipeline_snapshot.get("extra", {})
        if isinstance(pipeline_snapshot, dict)
        else {}
    )

    resolved_snapshot_path = str(
        extra.get("resolved_snapshot_path")
        or pipeline_extra.get("resolved_snapshot_path")
        or ""
    ).strip()
    resolved_snapshot_sha = str(
        extra.get("resolved_snapshot_sha256")
        or pipeline_extra.get("resolved_snapshot_sha256")
        or ""
    ).strip()
    if resolved_snapshot_path and not resolved_snapshot_sha:
        resolved_snapshot_candidate = _resolve_path(
            resolved_snapshot_path,
            repo_root=repo_root_path,
            run_dir=run_dir_path,
        )
        if resolved_snapshot_candidate.exists():
            resolved_snapshot_sha = _sha256(resolved_snapshot_candidate)

    excluded_tiles = _load_excluded_tiles(tile_policy_path)
    extra_excluded = extra.get("tile_excluded_shortnames", [])
    if isinstance(extra_excluded, list):
        for value in extra_excluded:
            text = str(value).strip()
            if text:
                excluded_tiles.append(text)
    seen_excluded: set[str] = set()
    excluded_tiles_ordered: list[str] = []
    for item in excluded_tiles:
        if item in seen_excluded:
            continue
        seen_excluded.add(item)
        excluded_tiles_ordered.append(item)

    tile_policy_sha = (
        _sha256(tile_policy_path)
        if tile_policy_path.exists()
        else str(extra.get("tile_exclusion_policy_sha256", "")).strip()
    )

    run_id = run_dir_path.name
    selection_id = f"{run_id}_{selected_patches_sha[:12]}"
    manifest = {
        "schema_version": "handoff_patch_format_v2",
        "selection_id": selection_id,
        "run_id": run_id,
        "run_dir": _rel_or_abs(run_dir_path, repo_root_path),
        "resolved_snapshot_path": resolved_snapshot_path,
        "resolved_snapshot_sha256": resolved_snapshot_sha,
        "patch_selection_csv_path": "selected_patches.csv",
        "patch_selection_csv_sha256": selected_patches_sha,
        "patch_selection_count": len(selected_rows),
        "patch_mask_requirements_path": "patch_mask_requirements.csv",
        "patch_mask_requirements_sha256": patch_mask_requirements_sha,
        "patch_split_manifest_path": "patch_split_manifest.json",
        "patch_split_manifest_sha256": split_manifest_out_sha,
        "patch_quicklook_root": "quicklooks",
        "patch_quicklook_format": "geotiff_deflate_rgb",
        "patch_quicklook_count": len(selected_rows),
        "tile_exclusion_policy_path": (
            _rel_or_abs(tile_policy_path, repo_root_path)
            if tile_policy_path.exists()
            else ""
        ),
        "tile_exclusion_policy_sha256": tile_policy_sha,
        "excluded_tiles": excluded_tiles_ordered,
        "split_authority": "masterarbeit_strassenerkennung_cv",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_patch_manifest_csv": _rel_or_abs(
            source_patch_manifest_csv, repo_root_path
        ),
        "source_patch_manifest_csv_sha256": source_patch_manifest_sha,
        "source_patch_manifest_resolution": source_patch_manifest_resolution,
        "source_patch_split_manifest_json": _rel_or_abs(
            source_patch_split_manifest,
            repo_root_path,
        ),
        "source_patch_split_manifest_sha256": source_patch_split_sha,
        "source_patch_split_manifest_resolution": source_patch_split_resolution,
        "source_images_dir": _rel_or_abs(images_dir_path, repo_root_path),
    }

    patch_handoff_manifest_path = out_dir_path / "patch_handoff_manifest.json"
    patch_handoff_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "run_dir": str(run_dir_path),
        "handoff_dir": str(out_dir_path),
        "selection_source": source_patch_manifest_resolution,
        "selection_count": len(selected_rows),
        "selection_id": selection_id,
        "patch_handoff_manifest_path": str(patch_handoff_manifest_path),
        "selected_patches_path": str(selected_patches_path),
        "patch_mask_requirements_path": str(patch_mask_requirements_path),
        "patch_split_manifest_path": str(split_manifest_out_path),
    }


def verify_tile_handoff(
    *,
    handoff_dir: str | Path,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    tile_exclusion_policy: str | Path = DEFAULT_TILE_POLICY,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = (
        Path(repo_root).resolve() if repo_root is not None else _repo_root()
    )
    handoff_dir_path = _resolve_path(handoff_dir, repo_root=repo_root_path).resolve()
    images_dir_path = _resolve_path(
        images_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    cli_policy_path = _resolve_path(
        tile_exclusion_policy,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()

    selected_path = handoff_dir_path / "selected_maps.csv"
    manifest_path = handoff_dir_path / "handoff_manifest.json"
    mask_requirements_path = handoff_dir_path / "mask_requirements.csv"

    schema_errors: list[str] = []
    for required_path in (selected_path, manifest_path, mask_requirements_path):
        if not required_path.exists():
            schema_errors.append(f"[SCHEMA] missing file: {required_path}")
    if schema_errors:
        raise HandoffCheckError(EXIT_SCHEMA, schema_errors)

    selected_rows, selected_fields = _load_csv(selected_path)
    mask_rows, mask_fields = _load_csv(mask_requirements_path)

    for col in REQUIRED_SELECTED_COLUMNS:
        if col not in selected_fields:
            schema_errors.append(f"[SCHEMA] selected_maps.csv missing column: {col}")
    for col in REQUIRED_MASK_COLUMNS:
        if col not in mask_fields:
            schema_errors.append(
                f"[SCHEMA] mask_requirements.csv missing column: {col}"
            )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        schema_errors.append(f"[SCHEMA] failed to parse handoff_manifest.json: {exc}")
        manifest = {}

    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            schema_errors.append(
                f"[SCHEMA] handoff_manifest.json missing field: {field}"
            )

    if manifest.get("split_authority") != "masterarbeit_strassenerkennung_cv":
        schema_errors.append(
            "[SCHEMA] handoff_manifest.json split_authority must be 'masterarbeit_strassenerkennung_cv'"
        )

    if manifest.get("selection_count") is not None:
        try:
            if int(manifest["selection_count"]) != len(selected_rows):
                schema_errors.append(
                    "[SCHEMA] selection_count in handoff_manifest.json does not match selected_maps.csv"
                )
        except Exception:
            schema_errors.append(
                "[SCHEMA] selection_count in handoff_manifest.json is not an integer"
            )

    actual_selected_sha = _sha256(selected_path)
    manifest_selected_sha = str(manifest.get("selection_csv_sha256", "")).strip()
    if manifest_selected_sha and manifest_selected_sha != actual_selected_sha:
        schema_errors.append(
            "[SCHEMA] selection_csv_sha256 mismatch for selected_maps.csv"
        )

    mask_requirements = {
        str(row.get("shortName", ""))
        .strip(): str(row.get("required_mask_filename", ""))
        .strip()
        for row in mask_rows
    }
    seen_short_names: set[str] = set()
    for idx, row in enumerate(selected_rows):
        short_name = str(row.get("shortName", "")).strip()
        if not short_name:
            schema_errors.append(
                f"[SCHEMA] selected_maps.csv row {idx} has empty shortName"
            )
            continue
        if short_name in seen_short_names:
            schema_errors.append(
                f"[SCHEMA] selected_maps.csv has duplicate shortName: {short_name}"
            )
        seen_short_names.add(short_name)
        if short_name not in mask_requirements:
            schema_errors.append(
                f"[SCHEMA] mask_requirements.csv missing shortName: {short_name}"
            )

    if schema_errors:
        raise HandoffCheckError(EXIT_SCHEMA, schema_errors)

    manifest_policy_path = str(manifest.get("tile_exclusion_policy_path", "")).strip()
    manifest_policy_resolved = (
        _resolve_path(manifest_policy_path, repo_root=repo_root_path)
        if manifest_policy_path
        else cli_policy_path
    )

    excluded_from_policy = _load_excluded_tiles(manifest_policy_resolved)
    excluded_from_manifest = [
        str(v).strip() for v in manifest.get("excluded_tiles", []) if str(v).strip()
    ]
    excluded_union = set(excluded_from_policy) | set(excluded_from_manifest)

    violations = sorted(excluded_union.intersection(seen_short_names))
    if violations:
        raise HandoffCheckError(
            EXIT_POLICY,
            [
                f"[POLICY] excluded tile present in selected_maps.csv: {item}"
                for item in violations
            ],
        )

    missing_images: list[str] = []
    missing_sidecars: list[str] = []

    for row in selected_rows:
        short_name = str(row.get("shortName", "")).strip()
        image_path_raw = str(row.get("image_path", "")).strip()
        image_filename = str(row.get("image_filename", "")).strip()
        candidates: list[Path] = []
        if image_path_raw:
            image_path = Path(image_path_raw)
            candidates.append(
                image_path if image_path.is_absolute() else repo_root_path / image_path
            )
        if image_filename:
            candidates.append(images_dir_path / image_filename)

        resolved_image = None
        seen_candidates: set[Path] = set()
        for candidate in candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            if candidate.exists():
                resolved_image = candidate
                break
        if resolved_image is None:
            missing_images.append(short_name)
            continue
        if resolved_image.suffix.lower() == ".png":
            sidecar = Path(str(resolved_image) + ".aux.xml")
            if not sidecar.exists():
                missing_sidecars.append(sidecar.name)

    if missing_images:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] image missing for shortName={short_name}"
                for short_name in missing_images
            ],
        )
    if missing_sidecars:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] missing PNG sidecar: {sidecar_name}"
                for sidecar_name in missing_sidecars
            ],
        )

    return {
        "status": "ok",
        "handoff_dir": str(handoff_dir_path),
        "selection_count": len(selected_rows),
        "excluded_tiles_checked": len(excluded_union),
    }


def verify_patch_handoff(
    *,
    handoff_dir: str | Path,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    tile_exclusion_policy: str | Path = DEFAULT_TILE_POLICY,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import rasterio  # type: ignore
    except Exception as exc:  # pragma: no cover - env issue
        raise HandoffCheckError(
            EXIT_SCHEMA,
            [f"[SCHEMA] rasterio is required for GeoTIFF patch verification: {exc}"],
        ) from exc

    repo_root_path = (
        Path(repo_root).resolve() if repo_root is not None else _repo_root()
    )
    handoff_dir_path = _resolve_path(handoff_dir, repo_root=repo_root_path).resolve()
    images_dir_path = _resolve_path(
        images_dir,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()
    cli_policy_path = _resolve_path(
        tile_exclusion_policy,
        repo_root=repo_root_path,
        prefer_repo=True,
    ).resolve()

    selected_path = handoff_dir_path / "selected_patches.csv"
    manifest_path = handoff_dir_path / "patch_handoff_manifest.json"
    mask_requirements_path = handoff_dir_path / "patch_mask_requirements.csv"
    split_manifest_path = handoff_dir_path / "patch_split_manifest.json"

    schema_errors: list[str] = []
    for required_path in (
        selected_path,
        manifest_path,
        mask_requirements_path,
        split_manifest_path,
    ):
        if not required_path.exists():
            schema_errors.append(f"[SCHEMA] missing file: {required_path}")
    if schema_errors:
        raise HandoffCheckError(EXIT_SCHEMA, schema_errors)

    selected_rows, selected_fields = _load_csv(selected_path)
    mask_rows, mask_fields = _load_csv(mask_requirements_path)
    for col in REQUIRED_PATCH_SELECTED_COLUMNS:
        if col not in selected_fields:
            schema_errors.append(f"[SCHEMA] selected_patches.csv missing column: {col}")
    for col in REQUIRED_PATCH_MASK_COLUMNS:
        if col not in mask_fields:
            schema_errors.append(
                f"[SCHEMA] patch_mask_requirements.csv missing column: {col}"
            )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        schema_errors.append(
            f"[SCHEMA] failed to parse patch_handoff_manifest.json: {exc}"
        )
        manifest = {}
    for field in REQUIRED_PATCH_MANIFEST_FIELDS:
        if field not in manifest:
            schema_errors.append(
                f"[SCHEMA] patch_handoff_manifest.json missing field: {field}"
            )

    if manifest.get("schema_version") != "handoff_patch_format_v2":
        schema_errors.append(
            "[SCHEMA] patch_handoff_manifest.json schema_version must be 'handoff_patch_format_v2'"
        )
    if manifest.get("patch_quicklook_format") != "geotiff_deflate_rgb":
        schema_errors.append(
            "[SCHEMA] patch_handoff_manifest.json patch_quicklook_format must be 'geotiff_deflate_rgb'"
        )
    if manifest.get("split_authority") != "masterarbeit_strassenerkennung_cv":
        schema_errors.append(
            "[SCHEMA] patch_handoff_manifest.json split_authority must be 'masterarbeit_strassenerkennung_cv'"
        )

    if manifest.get("patch_selection_count") is not None:
        try:
            if int(manifest["patch_selection_count"]) != len(selected_rows):
                schema_errors.append(
                    "[SCHEMA] patch_selection_count in patch_handoff_manifest.json does not match selected_patches.csv"
                )
        except Exception:
            schema_errors.append(
                "[SCHEMA] patch_selection_count in patch_handoff_manifest.json is not an integer"
            )

    actual_selected_sha = _sha256(selected_path)
    if str(manifest.get("patch_selection_csv_sha256", "")).strip() not in {
        "",
        actual_selected_sha,
    }:
        schema_errors.append(
            "[SCHEMA] patch_selection_csv_sha256 mismatch for selected_patches.csv"
        )
    actual_mask_sha = _sha256(mask_requirements_path)
    if str(manifest.get("patch_mask_requirements_sha256", "")).strip() not in {
        "",
        actual_mask_sha,
    }:
        schema_errors.append(
            "[SCHEMA] patch_mask_requirements_sha256 mismatch for patch_mask_requirements.csv"
        )
    actual_split_sha = _sha256(split_manifest_path)
    if str(manifest.get("patch_split_manifest_sha256", "")).strip() not in {
        "",
        actual_split_sha,
    }:
        schema_errors.append(
            "[SCHEMA] patch_split_manifest_sha256 mismatch for patch_split_manifest.json"
        )

    try:
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        schema_errors.append(
            f"[SCHEMA] failed to parse patch_split_manifest.json: {exc}"
        )
        split_manifest = {}
    patch_to_fold = (
        split_manifest.get("patch_to_fold", {})
        if isinstance(split_manifest, dict)
        else {}
    )
    if not isinstance(patch_to_fold, dict):
        schema_errors.append(
            "[SCHEMA] patch_split_manifest.json patch_to_fold must be an object"
        )
        patch_to_fold = {}

    mask_requirements = {
        str(row.get("patch_id", ""))
        .strip(): str(row.get("required_mask_filename", ""))
        .strip()
        for row in mask_rows
    }

    seen_patch_ids: set[str] = set()
    tile_to_folds: dict[str, set[int]] = {}
    for idx, row in enumerate(selected_rows):
        patch_id = str(row.get("patch_id", "")).strip()
        tile_shortname = str(row.get("tile_shortname", "")).strip()
        if not patch_id:
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv row {idx} has empty patch_id"
            )
            continue
        if patch_id in seen_patch_ids:
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv has duplicate patch_id: {patch_id}"
            )
            continue
        seen_patch_ids.add(patch_id)
        if patch_id not in mask_requirements:
            schema_errors.append(
                f"[SCHEMA] patch_mask_requirements.csv missing patch_id: {patch_id}"
            )
        if not tile_shortname:
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv row {idx} has empty tile_shortname for patch_id={patch_id}"
            )

        try:
            x0 = int(float(str(row.get("x0", "")).strip()))
            y0 = int(float(str(row.get("y0", "")).strip()))
            x1 = int(float(str(row.get("x1", "")).strip()))
            y1 = int(float(str(row.get("y1", "")).strip()))
        except Exception:
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv invalid bounds for patch_id={patch_id}"
            )
            continue
        if not (x1 > x0 and y1 > y0):
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv non-positive patch bounds for patch_id={patch_id}"
            )

        split_fold_raw = str(row.get("split_fold", "")).strip()
        try:
            split_fold = int(float(split_fold_raw))
        except Exception:
            schema_errors.append(
                f"[SCHEMA] selected_patches.csv invalid split_fold for patch_id={patch_id}"
            )
            continue

        fold_from_manifest = patch_to_fold.get(patch_id)
        if fold_from_manifest is None:
            schema_errors.append(
                f"[SCHEMA] patch_split_manifest.json missing patch_id in patch_to_fold: {patch_id}"
            )
        else:
            try:
                if int(fold_from_manifest) != split_fold:
                    schema_errors.append(
                        "[SCHEMA] split_fold mismatch for patch_id={}: selected_patches={}, split_manifest={}".format(
                            patch_id, split_fold, fold_from_manifest
                        )
                    )
            except Exception:
                schema_errors.append(
                    f"[SCHEMA] patch_split_manifest.json invalid fold for patch_id={patch_id}"
                )

        tile_to_folds.setdefault(tile_shortname, set()).add(split_fold)

    leaky_tiles = sorted(
        tile for tile, folds in tile_to_folds.items() if len(folds) > 1
    )
    if leaky_tiles:
        schema_errors.extend(
            [
                f"[SCHEMA] tile leakage across folds detected for tile_shortname={tile}"
                for tile in leaky_tiles
            ]
        )

    if schema_errors:
        raise HandoffCheckError(EXIT_SCHEMA, schema_errors)

    manifest_policy_path = str(manifest.get("tile_exclusion_policy_path", "")).strip()
    manifest_policy_resolved = (
        _resolve_path(manifest_policy_path, repo_root=repo_root_path)
        if manifest_policy_path
        else cli_policy_path
    )

    excluded_from_policy = _load_excluded_tiles(manifest_policy_resolved)
    excluded_from_manifest = [
        str(v).strip() for v in manifest.get("excluded_tiles", []) if str(v).strip()
    ]
    excluded_union = set(excluded_from_policy) | set(excluded_from_manifest)

    selected_tiles = {
        str(row.get("tile_shortname", "")).strip() for row in selected_rows
    }
    violations = sorted(excluded_union.intersection(selected_tiles))
    if violations:
        raise HandoffCheckError(
            EXIT_POLICY,
            [
                f"[POLICY] excluded tile present in selected_patches.csv: {item}"
                for item in violations
            ],
        )

    missing_images: list[str] = []
    missing_sidecars: list[str] = []
    missing_quicklooks: list[str] = []
    invalid_quicklook_format: list[str] = []
    invalid_quicklook_georef: list[str] = []

    for row in selected_rows:
        patch_id = str(row.get("patch_id", "")).strip()
        image_path_raw = str(row.get("image_path", "")).strip()
        image_filename = str(row.get("image_filename", "")).strip()

        candidates: list[Path] = []
        if image_path_raw:
            image_path = Path(image_path_raw)
            candidates.append(
                image_path if image_path.is_absolute() else repo_root_path / image_path
            )
        if image_filename:
            candidates.append(images_dir_path / image_filename)

        resolved_image = None
        seen_candidates: set[Path] = set()
        for candidate in candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            if candidate.exists():
                resolved_image = candidate
                break
        if resolved_image is None:
            missing_images.append(patch_id)
            continue
        if resolved_image.suffix.lower() == ".png":
            sidecar = Path(str(resolved_image) + ".aux.xml")
            if not sidecar.exists():
                missing_sidecars.append(sidecar.name)

        quicklook_path_raw = str(row.get("quicklook_path", "")).strip()
        if not quicklook_path_raw:
            missing_quicklooks.append(f"{patch_id}:<empty>")
            continue
        quicklook_path = Path(quicklook_path_raw)
        resolved_quicklook = (
            quicklook_path
            if quicklook_path.is_absolute()
            else (handoff_dir_path / quicklook_path).resolve()
        )
        if not resolved_quicklook.exists():
            missing_quicklooks.append(f"{patch_id}:{resolved_quicklook}")
            continue
        if resolved_quicklook.suffix.lower() not in {".tif", ".tiff"}:
            invalid_quicklook_format.append(
                f"{patch_id}:{resolved_quicklook} (expected .tif/.tiff)"
            )
            continue
        try:
            with rasterio.open(resolved_quicklook) as quicklook_ds:
                has_crs = quicklook_ds.crs is not None
                transform = quicklook_ds.transform
                has_transform = transform is not None and not transform.is_identity
                if not has_crs or not has_transform:
                    invalid_quicklook_georef.append(
                        f"{patch_id}:{resolved_quicklook} (missing embedded CRS/transform)"
                    )
        except Exception as exc:
            invalid_quicklook_georef.append(
                f"{patch_id}:{resolved_quicklook} (failed to read GeoTIFF: {exc})"
            )

    if missing_images:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] image missing for patch_id={patch_id}"
                for patch_id in missing_images
            ],
        )
    if missing_sidecars:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] missing PNG sidecar: {sidecar_name}"
                for sidecar_name in missing_sidecars
            ],
        )
    if missing_quicklooks:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] quicklook missing for patch_id={item}"
                for item in missing_quicklooks
            ],
        )
    if invalid_quicklook_format:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] quicklook format invalid for patch_id={item}"
                for item in invalid_quicklook_format
            ],
        )
    if invalid_quicklook_georef:
        raise HandoffCheckError(
            EXIT_DATA,
            [
                f"[DATA] quicklook georeference invalid for patch_id={item}"
                for item in invalid_quicklook_georef
            ],
        )

    return {
        "status": "ok",
        "handoff_dir": str(handoff_dir_path),
        "selection_count": len(selected_rows),
        "excluded_tiles_checked": len(excluded_union),
        "unique_tiles": len(selected_tiles),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handoff_check.sh")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--selection-csv", default="")
    prepare.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    prepare.add_argument("--tile-exclusion-policy", default=DEFAULT_TILE_POLICY)
    prepare.add_argument("--repo-root", default=str(_repo_root()))

    verify_local = subparsers.add_parser("verify-local")
    verify_local.add_argument("--handoff-dir", required=True)
    verify_local.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    verify_local.add_argument("--tile-exclusion-policy", default=DEFAULT_TILE_POLICY)
    verify_local.add_argument("--repo-root", default=str(_repo_root()))

    prepare_patches = subparsers.add_parser("prepare-patches")
    prepare_patches.add_argument("--run-dir", required=True)
    prepare_patches.add_argument("--out", required=True)
    prepare_patches.add_argument("--patch-manifest-csv", default="")
    prepare_patches.add_argument("--patch-split-manifest", default="")
    prepare_patches.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    prepare_patches.add_argument("--tile-exclusion-policy", default=DEFAULT_TILE_POLICY)
    prepare_patches.add_argument("--repo-root", default=str(_repo_root()))

    verify_patches = subparsers.add_parser("verify-patches")
    verify_patches.add_argument("--handoff-dir", required=True)
    verify_patches.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    verify_patches.add_argument("--tile-exclusion-policy", default=DEFAULT_TILE_POLICY)
    verify_patches.add_argument("--repo-root", default=str(_repo_root()))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    try:
        if ns.subcommand == "prepare":
            result = prepare_tile_handoff(
                run_dir=ns.run_dir,
                out_dir=ns.out,
                selection_csv=ns.selection_csv,
                images_dir=ns.images_dir,
                tile_exclusion_policy=ns.tile_exclusion_policy,
                repo_root=ns.repo_root,
            )
        elif ns.subcommand == "verify-local":
            result = verify_tile_handoff(
                handoff_dir=ns.handoff_dir,
                images_dir=ns.images_dir,
                tile_exclusion_policy=ns.tile_exclusion_policy,
                repo_root=ns.repo_root,
            )
        elif ns.subcommand == "prepare-patches":
            result = prepare_patch_handoff(
                run_dir=ns.run_dir,
                out_dir=ns.out,
                patch_manifest_csv=ns.patch_manifest_csv,
                patch_split_manifest=ns.patch_split_manifest,
                images_dir=ns.images_dir,
                tile_exclusion_policy=ns.tile_exclusion_policy,
                repo_root=ns.repo_root,
            )
        elif ns.subcommand == "verify-patches":
            result = verify_patch_handoff(
                handoff_dir=ns.handoff_dir,
                images_dir=ns.images_dir,
                tile_exclusion_policy=ns.tile_exclusion_policy,
                repo_root=ns.repo_root,
            )
        else:  # pragma: no cover - argparse enforces this
            parser.error(f"Unknown subcommand: {ns.subcommand}")
            return 1
    except HandoffCheckError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return exc.exit_code

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
