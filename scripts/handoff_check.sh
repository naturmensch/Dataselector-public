#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_IMAGES_DIR="data/images"
DEFAULT_TILE_POLICY="config/tile_exclusion_policy.yaml"

usage() {
  cat <<'USAGE'
Usage:
  scripts/handoff_check.sh prepare \
    --run-dir <outputs/runs/...> \
    --out <handoff_dir> \
    [--selection-csv <path>] \
    [--images-dir <path>] \
    [--tile-exclusion-policy <path>] \
    [--repo-root <path>]

  scripts/handoff_check.sh verify-local \
    --handoff-dir <path> \
    [--images-dir <path>] \
    [--tile-exclusion-policy <path>] \
    [--repo-root <path>]

  scripts/handoff_check.sh prepare-patches \
    --run-dir <outputs/runs/...> \
    --out <handoff_dir> \
    [--patch-manifest-csv <path>] \
    [--patch-split-manifest <path>] \
    [--images-dir <path>] \
    [--tile-exclusion-policy <path>] \
    [--repo-root <path>]

  scripts/handoff_check.sh verify-patches \
    --handoff-dir <path> \
    [--images-dir <path>] \
    [--tile-exclusion-policy <path>] \
    [--repo-root <path>]
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

subcommand="$1"
shift

case "$subcommand" in
  prepare)
    run_dir=""
    out_dir=""
    selection_csv=""
    images_dir="$DEFAULT_IMAGES_DIR"
    tile_policy="$DEFAULT_TILE_POLICY"
    repo_root="$DEFAULT_REPO_ROOT"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --run-dir)
          run_dir="${2:-}"
          shift 2
          ;;
        --out)
          out_dir="${2:-}"
          shift 2
          ;;
        --selection-csv)
          selection_csv="${2:-}"
          shift 2
          ;;
        --images-dir)
          images_dir="${2:-}"
          shift 2
          ;;
        --tile-exclusion-policy)
          tile_policy="${2:-}"
          shift 2
          ;;
        --repo-root)
          repo_root="${2:-}"
          shift 2
          ;;
        --help|-h)
          usage
          exit 0
          ;;
        *)
          echo "Unknown argument for prepare: $1" >&2
          usage
          exit 1
          ;;
      esac
    done

    if [[ -z "$run_dir" || -z "$out_dir" ]]; then
      echo "prepare requires --run-dir and --out" >&2
      exit 1
    fi

    RUN_DIR="$run_dir" \
    OUT_DIR="$out_dir" \
    EXPLICIT_SELECTION_CSV="$selection_csv" \
    IMAGES_DIR="$images_dir" \
    TILE_POLICY_PATH="$tile_policy" \
    REPO_ROOT="$repo_root" \
    python - <<'PY'
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path_str: str, *, repo_root: Path, run_dir: Path | None = None, prefer_repo: bool = False) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    if run_dir is not None:
        in_run = (run_dir / candidate)
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
    # stable de-dup order
    seen = set()
    ordered = []
    for item in excluded:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_selection_csv(
    *,
    run_dir: Path,
    explicit_selection_csv: str,
) -> tuple[Path, str]:
    if explicit_selection_csv.strip():
        path = _resolve_path(explicit_selection_csv.strip(), repo_root=repo_root, run_dir=run_dir, prefer_repo=True)
        if not path.exists():
            raise FileNotFoundError(f"explicit selection CSV not found: {path}")
        return path, "explicit"

    report_path = run_dir / "THESIS_PIPELINE_REPORT.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Selection file:\s*`([^`]+)`", text)
        if match:
            raw_path = match.group(1).strip()
            if raw_path and raw_path.lower() != "not available":
                candidate = (run_dir / raw_path).resolve()
                if candidate.exists():
                    return candidate, "report"

    meta_path = run_dir / "tuning_weights" / "meta.json"
    tuning_dir = run_dir / "tuning_weights"
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

    fallback = sorted(tuning_dir.glob("selection_a*_b*_g*.csv")) if tuning_dir.exists() else []
    if fallback:
        return fallback[0], "tuning_weights_fallback"

    raise FileNotFoundError(
        "could not resolve selection CSV (explicit path, report, and tuning fallback failed)"
    )


repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
run_dir = _resolve_path(os.environ["RUN_DIR"], repo_root=repo_root, prefer_repo=True).resolve()
out_dir = _resolve_path(os.environ["OUT_DIR"], repo_root=repo_root, prefer_repo=True).resolve()
explicit_selection_csv = os.environ.get("EXPLICIT_SELECTION_CSV", "")
images_dir = _resolve_path(os.environ.get("IMAGES_DIR", "data/images"), repo_root=repo_root, prefer_repo=True).resolve()
tile_policy_path = _resolve_path(
    os.environ.get("TILE_POLICY_PATH", "config/tile_exclusion_policy.yaml"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()

if not run_dir.exists():
    print(f"[SCHEMA] run directory not found: {run_dir}", file=sys.stderr)
    sys.exit(2)

out_dir.mkdir(parents=True, exist_ok=True)

try:
    source_selection_csv, source_selection = _resolve_selection_csv(
        run_dir=run_dir,
        explicit_selection_csv=explicit_selection_csv,
    )
except Exception as exc:
    print(f"[SCHEMA] {exc}", file=sys.stderr)
    sys.exit(2)

try:
    with source_selection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
        source_fields = list(reader.fieldnames or [])
except Exception as exc:
    print(f"[SCHEMA] failed to read selection CSV: {exc}", file=sys.stderr)
    sys.exit(2)

if not source_rows:
    print("[SCHEMA] selection CSV has no rows", file=sys.stderr)
    sys.exit(2)

optional_fields = ["year", "city", "city_source", "longName", "center_x", "center_y"]
normalized_rows: list[dict[str, str | int]] = []
errors: list[str] = []

for idx, row in enumerate(source_rows):
    short_name = str(row.get("shortName", "")).strip()
    image_filename = str(row.get("image_filename", "") or row.get("filename", "")).strip()
    image_path = str(row.get("image_path", "")).strip()

    if not short_name:
        fallback_name = image_filename or image_path
        if fallback_name:
            short_name = Path(fallback_name).stem

    if not short_name:
        errors.append(f"row {idx}: missing shortName and no filename fallback")
        continue

    if not image_filename:
        if image_path:
            image_filename = Path(image_path).name
        else:
            image_filename = f"{short_name}.png"

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
    for err in errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(2)

normalized_rows.sort(key=lambda item: (int(item["selection_rank"]), int(item["_order"])))

seen_short_names = set()
duplicates: list[str] = []
for row in normalized_rows:
    short_name = str(row["shortName"])
    if short_name in seen_short_names:
        duplicates.append(short_name)
    seen_short_names.add(short_name)

if duplicates:
    for item in sorted(set(duplicates)):
        print(f"[SCHEMA] duplicate shortName in selection: {item}", file=sys.stderr)
    sys.exit(2)

present_optional = [
    field for field in optional_fields if any(str(row.get(field, "")).strip() for row in normalized_rows)
]
selected_fieldnames = [
    "shortName",
    "image_path",
    "image_filename",
    "selection_rank",
] + present_optional

selected_maps_path = out_dir / "selected_maps.csv"
with selected_maps_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=selected_fieldnames)
    writer.writeheader()
    for row in normalized_rows:
        out_row = {field: row.get(field, "") for field in selected_fieldnames}
        writer.writerow(out_row)

mask_requirements_path = out_dir / "mask_requirements.csv"
with mask_requirements_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["shortName", "required_mask_filename"])
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

run_metadata_path = run_dir / "run_metadata.json"
run_metadata: dict = {}
if run_metadata_path.exists():
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except Exception:
        run_metadata = {}

extra = run_metadata.get("extra", {}) if isinstance(run_metadata, dict) else {}
pipeline_snapshot = extra.get("pipeline_metadata_snapshot", {}) if isinstance(extra, dict) else {}
pipeline_extra = pipeline_snapshot.get("extra", {}) if isinstance(pipeline_snapshot, dict) else {}

resolved_snapshot_path = str(extra.get("resolved_snapshot_path") or pipeline_extra.get("resolved_snapshot_path") or "").strip()
resolved_snapshot_sha = str(extra.get("resolved_snapshot_sha256") or pipeline_extra.get("resolved_snapshot_sha256") or "").strip()

if resolved_snapshot_path and not resolved_snapshot_sha:
    resolved_snapshot_candidate = _resolve_path(
        resolved_snapshot_path,
        repo_root=repo_root,
        run_dir=run_dir,
    )
    if resolved_snapshot_candidate.exists():
        resolved_snapshot_sha = _sha256(resolved_snapshot_candidate)

excluded_tiles = _load_excluded_tiles(tile_policy_path)
tile_policy_sha = _sha256(tile_policy_path) if tile_policy_path.exists() else ""

run_id = run_dir.name
selection_id = f"{run_id}_{selected_maps_sha[:12]}"

manifest = {
    "schema_version": "handoff_format_v1",
    "selection_id": selection_id,
    "run_id": run_id,
    "run_dir": _rel_or_abs(run_dir, repo_root),
    "resolved_snapshot_path": resolved_snapshot_path,
    "resolved_snapshot_sha256": resolved_snapshot_sha,
    "selection_csv_path": "selected_maps.csv",
    "selection_csv_sha256": selected_maps_sha,
    "selection_count": len(normalized_rows),
    "tile_exclusion_policy_path": _rel_or_abs(tile_policy_path, repo_root)
    if tile_policy_path.exists()
    else "",
    "tile_exclusion_policy_sha256": tile_policy_sha,
    "excluded_tiles": excluded_tiles,
    "split_authority": "masterarbeit_strassenerkennung_cv",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_selection_csv": _rel_or_abs(source_selection_csv, repo_root),
    "source_selection_csv_sha256": source_selection_sha,
    "source_selection_resolution": source_selection,
}

handoff_manifest_path = out_dir / "handoff_manifest.json"
handoff_manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=True, indent=2),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "status": "ok",
            "run_dir": str(run_dir),
            "handoff_dir": str(out_dir),
            "selection_source": source_selection,
            "selection_count": len(normalized_rows),
            "selection_id": selection_id,
        },
        ensure_ascii=True,
    )
)
sys.exit(0)
PY
    ;;

  prepare-patches)
    run_dir=""
    out_dir=""
    patch_manifest_csv=""
    patch_split_manifest=""
    images_dir="$DEFAULT_IMAGES_DIR"
    tile_policy="$DEFAULT_TILE_POLICY"
    repo_root="$DEFAULT_REPO_ROOT"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --run-dir)
          run_dir="${2:-}"
          shift 2
          ;;
        --out)
          out_dir="${2:-}"
          shift 2
          ;;
        --patch-manifest-csv)
          patch_manifest_csv="${2:-}"
          shift 2
          ;;
        --patch-split-manifest)
          patch_split_manifest="${2:-}"
          shift 2
          ;;
        --images-dir)
          images_dir="${2:-}"
          shift 2
          ;;
        --tile-exclusion-policy)
          tile_policy="${2:-}"
          shift 2
          ;;
        --repo-root)
          repo_root="${2:-}"
          shift 2
          ;;
        --help|-h)
          usage
          exit 0
          ;;
        *)
          echo "Unknown argument for prepare-patches: $1" >&2
          usage
          exit 1
          ;;
      esac
    done

    if [[ -z "$run_dir" || -z "$out_dir" ]]; then
      echo "prepare-patches requires --run-dir and --out" >&2
      exit 1
    fi

    RUN_DIR="$run_dir" \
    OUT_DIR="$out_dir" \
    EXPLICIT_PATCH_MANIFEST_CSV="$patch_manifest_csv" \
    EXPLICIT_PATCH_SPLIT_MANIFEST="$patch_split_manifest" \
    IMAGES_DIR="$images_dir" \
    TILE_POLICY_PATH="$tile_policy" \
    REPO_ROOT="$repo_root" \
    python - <<'PY'
import csv
import hashlib
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(
    path_str: str,
    *,
    repo_root: Path,
    run_dir: Path | None = None,
    prefer_repo: bool = False,
) -> Path:
    candidate = Path(path_str)
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
    seen = set()
    ordered = []
    for item in excluded:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_patch_manifest_csv(
    *,
    run_dir: Path,
    explicit_path: str,
) -> tuple[Path, str]:
    if explicit_path.strip():
        path = _resolve_path(
            explicit_path.strip(),
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if not path.exists():
            raise FileNotFoundError(f"explicit patch manifest CSV not found: {path}")
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
                    artifact_rel, repo_root=repo_root, run_dir=run_dir
                ).resolve()
                if candidate.exists():
                    return candidate, "annotation_plan_contract"
        except Exception:
            pass

    raise FileNotFoundError(
        "could not resolve patch manifest CSV (explicit path and annotation_plan defaults failed)"
    )


def _resolve_patch_split_manifest(
    *,
    run_dir: Path,
    explicit_path: str,
) -> tuple[Path, str]:
    if explicit_path.strip():
        path = _resolve_path(
            explicit_path.strip(),
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"explicit patch split manifest not found: {path}"
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
                (contract.get("artifacts", {}) or {}).get("patch_split_manifest_json", "")
            ).strip()
            if artifact_rel:
                candidate = _resolve_path(
                    artifact_rel, repo_root=repo_root, run_dir=run_dir
                ).resolve()
                if candidate.exists():
                    return candidate, "annotation_plan_contract"
        except Exception:
            pass

    raise FileNotFoundError(
        "could not resolve patch split manifest (explicit path and annotation_plan defaults failed)"
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


def _find_aux_sidecar_for_image(image_path: Path) -> Path | None:
    candidates = [
        Path(str(image_path) + ".aux.xml"),
        image_path.with_suffix(image_path.suffix + ".aux.xml"),
        image_path.with_suffix(".aux.xml"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _parse_geotransform(raw: str) -> tuple[float, float, float, float, float, float] | None:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) < 6:
        return None
    try:
        values = tuple(float(p) for p in parts[:6])
    except Exception:
        return None
    return values  # type: ignore[return-value]


def _format_geotransform(gt: tuple[float, float, float, float, float, float]) -> str:
    return ", ".join(f"{value: .16e}" for value in gt)


def _write_shifted_patch_aux(
    *,
    source_aux: Path,
    out_path: Path,
    x0: int,
    y0: int,
) -> bool:
    try:
        tree = ET.parse(source_aux)
        root = tree.getroot()
        geot = root.find("GeoTransform")
        if geot is None:
            geot = root.find(".//GeoTransform")
        if geot is None or not geot.text:
            return False
        parsed = _parse_geotransform(geot.text)
        if parsed is None:
            return False
        gt0, gt1, gt2, gt3, gt4, gt5 = parsed
        shifted = (
            gt0 + (float(x0) * gt1) + (float(y0) * gt2),
            gt1,
            gt2,
            gt3 + (float(x0) * gt4) + (float(y0) * gt5),
            gt4,
            gt5,
        )
        geot.text = f"  {_format_geotransform(shifted)}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8")
        return True
    except Exception:
        return False


def _resolve_source_image_for_row(
    *,
    row: dict[str, str | int],
    repo_root: Path,
    run_dir: Path,
    images_dir: Path,
) -> Path | None:
    image_path_raw = str(row.get("image_path", "")).strip()
    image_filename = str(row.get("image_filename", "")).strip()
    candidates: list[Path] = []
    if image_path_raw:
        image_path = Path(image_path_raw)
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append((repo_root / image_path).resolve())
    if image_filename:
        candidates.append((images_dir / image_filename).resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
run_dir = _resolve_path(os.environ["RUN_DIR"], repo_root=repo_root, prefer_repo=True).resolve()
out_dir = _resolve_path(os.environ["OUT_DIR"], repo_root=repo_root, prefer_repo=True).resolve()
explicit_patch_manifest_csv = os.environ.get("EXPLICIT_PATCH_MANIFEST_CSV", "")
explicit_patch_split_manifest = os.environ.get("EXPLICIT_PATCH_SPLIT_MANIFEST", "")
images_dir = _resolve_path(
    os.environ.get("IMAGES_DIR", "data/images"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()
tile_policy_path = _resolve_path(
    os.environ.get("TILE_POLICY_PATH", "config/tile_exclusion_policy.yaml"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()

if not run_dir.exists():
    print(f"[SCHEMA] run directory not found: {run_dir}", file=sys.stderr)
    sys.exit(2)

out_dir.mkdir(parents=True, exist_ok=True)

try:
    source_patch_manifest_csv, source_patch_manifest_resolution = _resolve_patch_manifest_csv(
        run_dir=run_dir,
        explicit_path=explicit_patch_manifest_csv,
    )
    source_patch_split_manifest, source_patch_split_resolution = _resolve_patch_split_manifest(
        run_dir=run_dir,
        explicit_path=explicit_patch_split_manifest,
    )
except Exception as exc:
    print(f"[SCHEMA] {exc}", file=sys.stderr)
    sys.exit(2)

try:
    with source_patch_manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
        source_fields = list(reader.fieldnames or [])
except Exception as exc:
    print(f"[SCHEMA] failed to read patch_manifest.csv: {exc}", file=sys.stderr)
    sys.exit(2)

if not source_rows:
    print("[SCHEMA] patch_manifest.csv has no rows", file=sys.stderr)
    sys.exit(2)

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
for col in required_patch_columns:
    if col not in source_fields:
        print(f"[SCHEMA] patch_manifest.csv missing column: {col}", file=sys.stderr)
        sys.exit(2)

try:
    split_manifest = json.loads(
        source_patch_split_manifest.read_text(encoding="utf-8")
    )
except Exception as exc:
    print(f"[SCHEMA] failed to parse patch split manifest: {exc}", file=sys.stderr)
    sys.exit(2)

patch_to_fold = split_manifest.get("patch_to_fold", {})
if not isinstance(patch_to_fold, dict):
    print("[SCHEMA] patch split manifest missing object: patch_to_fold", file=sys.stderr)
    sys.exit(2)

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
        errors.append(f"row {idx}: missing patch_id")
        continue
    if patch_id in seen_patch_ids:
        errors.append(f"duplicate patch_id in patch_manifest.csv: {patch_id}")
        continue
    seen_patch_ids.add(patch_id)

    if not tile_shortname:
        errors.append(f"row {idx}: missing tile_shortname for patch_id={patch_id}")
        continue
    if not image_path:
        errors.append(f"row {idx}: missing image_path for patch_id={patch_id}")
        continue

    quicklook_path = str(row.get("quicklook_path", "")).strip()
    if not quicklook_path:
        errors.append(f"row {idx}: missing quicklook_path for patch_id={patch_id}")
        continue
    quicklook_aux_path = str(row.get("quicklook_aux_path", "")).strip()
    if not quicklook_aux_path:
        quicklook_aux_path = f"{quicklook_path}.aux.xml"

    quicklook_rel = Path(quicklook_path)
    quicklook_aux_rel = Path(quicklook_aux_path)
    if quicklook_rel.is_absolute() or ".." in quicklook_rel.parts:
        errors.append(f"row {idx}: invalid quicklook_path for patch_id={patch_id}")
        continue
    if quicklook_aux_rel.is_absolute() or ".." in quicklook_aux_rel.parts:
        errors.append(f"row {idx}: invalid quicklook_aux_path for patch_id={patch_id}")
        continue

    try:
        x0 = int(float(str(row.get("x0", "")).strip()))
        y0 = int(float(str(row.get("y0", "")).strip()))
        x1 = int(float(str(row.get("x1", "")).strip()))
        y1 = int(float(str(row.get("y1", "")).strip()))
    except Exception:
        errors.append(f"row {idx}: invalid patch bounds for patch_id={patch_id}")
        continue

    split_fold_raw = str(row.get("split_fold", "")).strip()
    if split_fold_raw:
        try:
            split_fold = int(float(split_fold_raw))
        except Exception:
            errors.append(f"row {idx}: invalid split_fold for patch_id={patch_id}")
            continue
    else:
        fold_from_manifest = patch_to_fold.get(patch_id)
        if fold_from_manifest is None:
            errors.append(
                f"row {idx}: split_fold missing and not found in patch split manifest for patch_id={patch_id}"
            )
            continue
        try:
            split_fold = int(fold_from_manifest)
        except Exception:
            errors.append(
                f"row {idx}: invalid patch_to_fold mapping for patch_id={patch_id}"
            )
            continue

    patch_index_raw = str(row.get("patch_index", "")).strip()
    try:
        patch_index = int(float(patch_index_raw)) if patch_index_raw else 0
    except Exception:
        patch_index = 0

    selection_rank_raw = str(row.get("selection_rank", "")).strip()
    try:
        selection_rank = int(float(selection_rank_raw)) if selection_rank_raw else idx
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
            "quicklook_aux_path": str(quicklook_aux_rel),
            "qc_status": str(row.get("qc_status", "")).strip(),
            "qc_reason": str(row.get("qc_reason", "")).strip(),
            "_order": idx,
        }
    )

if errors:
    for err in errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(2)

if not selected_rows:
    print("[SCHEMA] patch_manifest.csv has no qc_passed rows", file=sys.stderr)
    sys.exit(2)

selected_rows.sort(
    key=lambda item: (int(item["selection_rank"]), int(item["patch_index"]), int(item["_order"]))
)

quicklook_copy_errors: list[str] = []
for row in selected_rows:
    patch_id = str(row["patch_id"])
    quicklook_rel = str(row["quicklook_path"]).strip()
    quicklook_aux_rel = str(row["quicklook_aux_path"]).strip()

    quicklook_src = _resolve_manifest_artifact(
        artifact_path=quicklook_rel,
        source_manifest_path=source_patch_manifest_csv,
        run_dir=run_dir,
        repo_root=repo_root,
    ).resolve()
    quicklook_aux_src = _resolve_manifest_artifact(
        artifact_path=quicklook_aux_rel,
        source_manifest_path=source_patch_manifest_csv,
        run_dir=run_dir,
        repo_root=repo_root,
    ).resolve()

    if not quicklook_src.exists():
        quicklook_copy_errors.append(
            f"missing quicklook image for patch_id={patch_id}: {quicklook_src}"
        )
        continue
    if not quicklook_aux_src.exists():
        source_image = _resolve_source_image_for_row(
            row=row,
            repo_root=repo_root,
            run_dir=run_dir,
            images_dir=images_dir,
        )
        if source_image is None:
            quicklook_copy_errors.append(
                f"missing quicklook sidecar and could not resolve source image for patch_id={patch_id}"
            )
            continue
        source_aux = _find_aux_sidecar_for_image(source_image)
        if source_aux is None:
            quicklook_copy_errors.append(
                f"missing quicklook sidecar and source image sidecar for patch_id={patch_id}: {source_image}"
            )
            continue
        synthesized_aux_target = (out_dir / quicklook_aux_rel).resolve()
        if not _write_shifted_patch_aux(
            source_aux=source_aux,
            out_path=synthesized_aux_target,
            x0=int(row["x0"]),
            y0=int(row["y0"]),
        ):
            quicklook_copy_errors.append(
                f"missing quicklook sidecar and failed to synthesize from source sidecar for patch_id={patch_id}: {source_aux}"
            )
            continue

    quicklook_dst = (out_dir / quicklook_rel).resolve()
    quicklook_aux_dst = (out_dir / quicklook_aux_rel).resolve()
    quicklook_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(quicklook_src, quicklook_dst)
    if quicklook_aux_src.exists():
        quicklook_aux_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(quicklook_aux_src, quicklook_aux_dst)

if quicklook_copy_errors:
    for err in quicklook_copy_errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(2)

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
    "quicklook_aux_path",
    "qc_status",
    "qc_reason",
]

selected_patches_path = out_dir / "selected_patches.csv"
with selected_patches_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=selected_patches_fields)
    writer.writeheader()
    for row in selected_rows:
        out_row = {field: row.get(field, "") for field in selected_patches_fields}
        writer.writerow(out_row)

patch_mask_requirements_path = out_dir / "patch_mask_requirements.csv"
with patch_mask_requirements_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["patch_id", "required_mask_filename"])
    writer.writeheader()
    for row in selected_rows:
        patch_id = str(row["patch_id"])
        writer.writerow(
            {
                "patch_id": patch_id,
                "required_mask_filename": f"{patch_id}_mask.tif",
            }
        )

split_manifest_out_path = out_dir / "patch_split_manifest.json"
split_manifest_out_path.write_text(
    json.dumps(split_manifest, ensure_ascii=True, indent=2),
    encoding="utf-8",
)

selected_patches_sha = _sha256(selected_patches_path)
patch_mask_requirements_sha = _sha256(patch_mask_requirements_path)
source_patch_manifest_sha = _sha256(source_patch_manifest_csv)
source_patch_split_sha = _sha256(source_patch_split_manifest)
split_manifest_out_sha = _sha256(split_manifest_out_path)

run_metadata_path = run_dir / "run_metadata.json"
run_metadata: dict = {}
if run_metadata_path.exists():
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except Exception:
        run_metadata = {}

extra = run_metadata.get("extra", {}) if isinstance(run_metadata, dict) else {}
pipeline_snapshot = extra.get("pipeline_metadata_snapshot", {}) if isinstance(extra, dict) else {}
pipeline_extra = pipeline_snapshot.get("extra", {}) if isinstance(pipeline_snapshot, dict) else {}

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
        repo_root=repo_root,
        run_dir=run_dir,
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
seen_excluded = set()
excluded_tiles_ordered = []
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

run_id = run_dir.name
selection_id = f"{run_id}_{selected_patches_sha[:12]}"

manifest = {
    "schema_version": "handoff_patch_format_v1",
    "selection_id": selection_id,
    "run_id": run_id,
    "run_dir": _rel_or_abs(run_dir, repo_root),
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
    "patch_quicklook_count": len(selected_rows),
    "tile_exclusion_policy_path": _rel_or_abs(tile_policy_path, repo_root)
    if tile_policy_path.exists()
    else "",
    "tile_exclusion_policy_sha256": tile_policy_sha,
    "excluded_tiles": excluded_tiles_ordered,
    "split_authority": "masterarbeit_strassenerkennung_cv",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_patch_manifest_csv": _rel_or_abs(source_patch_manifest_csv, repo_root),
    "source_patch_manifest_csv_sha256": source_patch_manifest_sha,
    "source_patch_manifest_resolution": source_patch_manifest_resolution,
    "source_patch_split_manifest_json": _rel_or_abs(source_patch_split_manifest, repo_root),
    "source_patch_split_manifest_sha256": source_patch_split_sha,
    "source_patch_split_manifest_resolution": source_patch_split_resolution,
    "source_images_dir": _rel_or_abs(images_dir, repo_root),
}

patch_handoff_manifest_path = out_dir / "patch_handoff_manifest.json"
patch_handoff_manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=True, indent=2),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "status": "ok",
            "run_dir": str(run_dir),
            "handoff_dir": str(out_dir),
            "selection_source": source_patch_manifest_resolution,
            "selection_count": len(selected_rows),
            "selection_id": selection_id,
        },
        ensure_ascii=True,
    )
)
sys.exit(0)
PY
    ;;

  verify-patches)
    handoff_dir=""
    images_dir="$DEFAULT_IMAGES_DIR"
    tile_policy="$DEFAULT_TILE_POLICY"
    repo_root="$DEFAULT_REPO_ROOT"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --handoff-dir)
          handoff_dir="${2:-}"
          shift 2
          ;;
        --images-dir)
          images_dir="${2:-}"
          shift 2
          ;;
        --tile-exclusion-policy)
          tile_policy="${2:-}"
          shift 2
          ;;
        --repo-root)
          repo_root="${2:-}"
          shift 2
          ;;
        --help|-h)
          usage
          exit 0
          ;;
        *)
          echo "Unknown argument for verify-patches: $1" >&2
          usage
          exit 1
          ;;
      esac
    done

    if [[ -z "$handoff_dir" ]]; then
      echo "verify-patches requires --handoff-dir" >&2
      exit 1
    fi

    HANDOFF_DIR="$handoff_dir" \
    IMAGES_DIR="$images_dir" \
    TILE_POLICY_PATH="$tile_policy" \
    REPO_ROOT="$repo_root" \
    python - <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


EXIT_SCHEMA = 2
EXIT_DATA = 3
EXIT_POLICY = 4

REQUIRED_SELECTED_COLUMNS = [
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
    "quicklook_aux_path",
]
REQUIRED_MASK_COLUMNS = ["patch_id", "required_mask_filename"]
REQUIRED_MANIFEST_FIELDS = [
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
    "tile_exclusion_policy_path",
    "tile_exclusion_policy_sha256",
    "excluded_tiles",
    "split_authority",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path_str: str, *, repo_root: Path, prefer_repo: bool = False) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    in_repo = repo_root / candidate
    if prefer_repo or in_repo.exists():
        return in_repo
    return candidate


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


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
    seen = set()
    ordered = []
    for item in excluded:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
handoff_dir = _resolve_path(os.environ["HANDOFF_DIR"], repo_root=repo_root).resolve()
images_dir = _resolve_path(
    os.environ.get("IMAGES_DIR", "data/images"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()
cli_policy_path = _resolve_path(
    os.environ.get("TILE_POLICY_PATH", "config/tile_exclusion_policy.yaml"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()

selected_path = handoff_dir / "selected_patches.csv"
manifest_path = handoff_dir / "patch_handoff_manifest.json"
mask_requirements_path = handoff_dir / "patch_mask_requirements.csv"
split_manifest_path = handoff_dir / "patch_split_manifest.json"

schema_errors: list[str] = []
for required_path in (
    selected_path,
    manifest_path,
    mask_requirements_path,
    split_manifest_path,
):
    if not required_path.exists():
        schema_errors.append(f"missing file: {required_path}")

if schema_errors:
    for err in schema_errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(EXIT_SCHEMA)

selected_rows, selected_fields = _load_csv(selected_path)
mask_rows, mask_fields = _load_csv(mask_requirements_path)

for col in REQUIRED_SELECTED_COLUMNS:
    if col not in selected_fields:
        schema_errors.append(f"selected_patches.csv missing column: {col}")
for col in REQUIRED_MASK_COLUMNS:
    if col not in mask_fields:
        schema_errors.append(f"patch_mask_requirements.csv missing column: {col}")

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    schema_errors.append(f"failed to parse patch_handoff_manifest.json: {exc}")
    manifest = {}

for field in REQUIRED_MANIFEST_FIELDS:
    if field not in manifest:
        schema_errors.append(f"patch_handoff_manifest.json missing field: {field}")

if manifest.get("split_authority") != "masterarbeit_strassenerkennung_cv":
    schema_errors.append(
        "patch_handoff_manifest.json split_authority must be 'masterarbeit_strassenerkennung_cv'"
    )

if manifest.get("patch_selection_count") is not None:
    try:
        if int(manifest["patch_selection_count"]) != len(selected_rows):
            schema_errors.append(
                "patch_selection_count in patch_handoff_manifest.json does not match selected_patches.csv"
            )
    except Exception:
        schema_errors.append(
            "patch_selection_count in patch_handoff_manifest.json is not an integer"
        )

actual_selected_sha = _sha256(selected_path)
manifest_selected_sha = str(manifest.get("patch_selection_csv_sha256", "")).strip()
if manifest_selected_sha and manifest_selected_sha != actual_selected_sha:
    schema_errors.append("patch_selection_csv_sha256 mismatch for selected_patches.csv")

actual_mask_sha = _sha256(mask_requirements_path)
manifest_mask_sha = str(manifest.get("patch_mask_requirements_sha256", "")).strip()
if manifest_mask_sha and manifest_mask_sha != actual_mask_sha:
    schema_errors.append(
        "patch_mask_requirements_sha256 mismatch for patch_mask_requirements.csv"
    )

actual_split_sha = _sha256(split_manifest_path)
manifest_split_sha = str(manifest.get("patch_split_manifest_sha256", "")).strip()
if manifest_split_sha and manifest_split_sha != actual_split_sha:
    schema_errors.append("patch_split_manifest_sha256 mismatch for patch_split_manifest.json")

try:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    schema_errors.append(f"failed to parse patch_split_manifest.json: {exc}")
    split_manifest = {}
patch_to_fold = split_manifest.get("patch_to_fold", {}) if isinstance(split_manifest, dict) else {}
if not isinstance(patch_to_fold, dict):
    schema_errors.append("patch_split_manifest.json patch_to_fold must be an object")
    patch_to_fold = {}

mask_requirements = {
    str(row.get("patch_id", "")).strip(): str(
        row.get("required_mask_filename", "")
    ).strip()
    for row in mask_rows
}

seen_patch_ids = set()
tile_to_folds: dict[str, set[int]] = {}
for idx, row in enumerate(selected_rows):
    patch_id = str(row.get("patch_id", "")).strip()
    tile_shortname = str(row.get("tile_shortname", "")).strip()
    if not patch_id:
        schema_errors.append(f"selected_patches.csv row {idx} has empty patch_id")
        continue
    if patch_id in seen_patch_ids:
        schema_errors.append(f"selected_patches.csv has duplicate patch_id: {patch_id}")
        continue
    seen_patch_ids.add(patch_id)

    if patch_id not in mask_requirements:
        schema_errors.append(
            f"patch_mask_requirements.csv missing patch_id: {patch_id}"
        )

    if not tile_shortname:
        schema_errors.append(
            f"selected_patches.csv row {idx} has empty tile_shortname for patch_id={patch_id}"
        )

    try:
        x0 = int(float(str(row.get("x0", "")).strip()))
        y0 = int(float(str(row.get("y0", "")).strip()))
        x1 = int(float(str(row.get("x1", "")).strip()))
        y1 = int(float(str(row.get("y1", "")).strip()))
    except Exception:
        schema_errors.append(f"selected_patches.csv invalid bounds for patch_id={patch_id}")
        continue
    if not (x1 > x0 and y1 > y0):
        schema_errors.append(
            f"selected_patches.csv non-positive patch bounds for patch_id={patch_id}"
        )

    split_fold_raw = str(row.get("split_fold", "")).strip()
    try:
        split_fold = int(float(split_fold_raw))
    except Exception:
        schema_errors.append(f"selected_patches.csv invalid split_fold for patch_id={patch_id}")
        continue

    fold_from_manifest = patch_to_fold.get(patch_id)
    if fold_from_manifest is None:
        schema_errors.append(
            f"patch_split_manifest.json missing patch_id in patch_to_fold: {patch_id}"
        )
    else:
        try:
            if int(fold_from_manifest) != split_fold:
                schema_errors.append(
                    f"split_fold mismatch for patch_id={patch_id}: selected_patches={split_fold}, split_manifest={fold_from_manifest}"
                )
        except Exception:
            schema_errors.append(
                f"patch_split_manifest.json invalid fold for patch_id={patch_id}"
            )

    tile_to_folds.setdefault(tile_shortname, set()).add(split_fold)

leaky_tiles = sorted(tile for tile, folds in tile_to_folds.items() if len(folds) > 1)
if leaky_tiles:
    for tile in leaky_tiles:
        schema_errors.append(
            f"tile leakage across folds detected for tile_shortname={tile}"
        )

if schema_errors:
    for err in schema_errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(EXIT_SCHEMA)

manifest_policy_path = str(manifest.get("tile_exclusion_policy_path", "")).strip()
if manifest_policy_path:
    manifest_policy_resolved = _resolve_path(manifest_policy_path, repo_root=repo_root)
else:
    manifest_policy_resolved = cli_policy_path

excluded_from_policy = _load_excluded_tiles(manifest_policy_resolved)
excluded_from_manifest = [
    str(v).strip() for v in manifest.get("excluded_tiles", []) if str(v).strip()
]
excluded_union = set(excluded_from_policy) | set(excluded_from_manifest)

selected_tiles = {str(row.get("tile_shortname", "")).strip() for row in selected_rows}
violations = sorted(excluded_union.intersection(selected_tiles))
if violations:
    for item in violations:
        print(
            f"[POLICY] excluded tile present in selected_patches.csv: {item}",
            file=sys.stderr,
        )
    sys.exit(EXIT_POLICY)

missing_images: list[str] = []
missing_sidecars: list[str] = []
missing_quicklooks: list[str] = []
missing_quicklook_sidecars: list[str] = []

for row in selected_rows:
    patch_id = str(row.get("patch_id", "")).strip()
    image_path_raw = str(row.get("image_path", "")).strip()
    image_filename = str(row.get("image_filename", "")).strip()

    candidates: list[Path] = []
    if image_path_raw:
        image_path = Path(image_path_raw)
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append(repo_root / image_path)
    if image_filename:
        candidates.append(images_dir / image_filename)

    resolved_image = None
    seen_candidates = set()
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
    quicklook_aux_path_raw = str(row.get("quicklook_aux_path", "")).strip()
    if not quicklook_aux_path_raw and quicklook_path_raw:
        quicklook_aux_path_raw = f"{quicklook_path_raw}.aux.xml"

    if not quicklook_path_raw:
        missing_quicklooks.append(f"{patch_id}:<empty>")
    else:
        quicklook_path = Path(quicklook_path_raw)
        if quicklook_path.is_absolute():
            resolved_quicklook = quicklook_path
        else:
            resolved_quicklook = (handoff_dir / quicklook_path).resolve()
        if not resolved_quicklook.exists():
            missing_quicklooks.append(f"{patch_id}:{resolved_quicklook}")

    if not quicklook_aux_path_raw:
        missing_quicklook_sidecars.append(f"{patch_id}:<empty>")
    else:
        quicklook_aux_path = Path(quicklook_aux_path_raw)
        if quicklook_aux_path.is_absolute():
            resolved_quicklook_aux = quicklook_aux_path
        else:
            resolved_quicklook_aux = (handoff_dir / quicklook_aux_path).resolve()
        if not resolved_quicklook_aux.exists():
            missing_quicklook_sidecars.append(f"{patch_id}:{resolved_quicklook_aux}")

if missing_images:
    for patch_id in missing_images:
        print(f"[DATA] image missing for patch_id={patch_id}", file=sys.stderr)
    sys.exit(EXIT_DATA)

if missing_sidecars:
    for sidecar_name in missing_sidecars:
        print(f"[DATA] missing PNG sidecar: {sidecar_name}", file=sys.stderr)
    sys.exit(EXIT_DATA)

if missing_quicklooks:
    for item in missing_quicklooks:
        print(f"[DATA] quicklook missing for patch_id={item}", file=sys.stderr)
    sys.exit(EXIT_DATA)

if missing_quicklook_sidecars:
    for item in missing_quicklook_sidecars:
        print(f"[DATA] quicklook sidecar missing for patch_id={item}", file=sys.stderr)
    sys.exit(EXIT_DATA)

print(
    json.dumps(
        {
            "status": "ok",
            "handoff_dir": str(handoff_dir),
            "selection_count": len(selected_rows),
            "excluded_tiles_checked": len(excluded_union),
            "unique_tiles": len(selected_tiles),
        },
        ensure_ascii=True,
    )
)
sys.exit(0)
PY
    ;;

  verify-local)
    handoff_dir=""
    images_dir="$DEFAULT_IMAGES_DIR"
    tile_policy="$DEFAULT_TILE_POLICY"
    repo_root="$DEFAULT_REPO_ROOT"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --handoff-dir)
          handoff_dir="${2:-}"
          shift 2
          ;;
        --images-dir)
          images_dir="${2:-}"
          shift 2
          ;;
        --tile-exclusion-policy)
          tile_policy="${2:-}"
          shift 2
          ;;
        --repo-root)
          repo_root="${2:-}"
          shift 2
          ;;
        --help|-h)
          usage
          exit 0
          ;;
        *)
          echo "Unknown argument for verify-local: $1" >&2
          usage
          exit 1
          ;;
      esac
    done

    if [[ -z "$handoff_dir" ]]; then
      echo "verify-local requires --handoff-dir" >&2
      exit 1
    fi

    HANDOFF_DIR="$handoff_dir" \
    IMAGES_DIR="$images_dir" \
    TILE_POLICY_PATH="$tile_policy" \
    REPO_ROOT="$repo_root" \
    python - <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


EXIT_SCHEMA = 2
EXIT_DATA = 3
EXIT_POLICY = 4

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path_str: str, *, repo_root: Path, prefer_repo: bool = False) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    in_repo = repo_root / candidate
    if prefer_repo or in_repo.exists():
        return in_repo
    return candidate


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


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
    seen = set()
    ordered = []
    for item in excluded:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
handoff_dir = _resolve_path(os.environ["HANDOFF_DIR"], repo_root=repo_root).resolve()
images_dir = _resolve_path(os.environ.get("IMAGES_DIR", "data/images"), repo_root=repo_root, prefer_repo=True).resolve()
cli_policy_path = _resolve_path(
    os.environ.get("TILE_POLICY_PATH", "config/tile_exclusion_policy.yaml"),
    repo_root=repo_root,
    prefer_repo=True,
).resolve()

selected_path = handoff_dir / "selected_maps.csv"
manifest_path = handoff_dir / "handoff_manifest.json"
mask_requirements_path = handoff_dir / "mask_requirements.csv"

schema_errors: list[str] = []
for required_path in (selected_path, manifest_path, mask_requirements_path):
    if not required_path.exists():
        schema_errors.append(f"missing file: {required_path}")

if schema_errors:
    for err in schema_errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(EXIT_SCHEMA)

selected_rows, selected_fields = _load_csv(selected_path)
mask_rows, mask_fields = _load_csv(mask_requirements_path)

for col in REQUIRED_SELECTED_COLUMNS:
    if col not in selected_fields:
        schema_errors.append(f"selected_maps.csv missing column: {col}")
for col in REQUIRED_MASK_COLUMNS:
    if col not in mask_fields:
        schema_errors.append(f"mask_requirements.csv missing column: {col}")

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    schema_errors.append(f"failed to parse handoff_manifest.json: {exc}")
    manifest = {}

for field in REQUIRED_MANIFEST_FIELDS:
    if field not in manifest:
        schema_errors.append(f"handoff_manifest.json missing field: {field}")

if manifest.get("split_authority") != "masterarbeit_strassenerkennung_cv":
    schema_errors.append(
        "handoff_manifest.json split_authority must be 'masterarbeit_strassenerkennung_cv'"
    )

if manifest.get("selection_count") is not None:
    try:
        if int(manifest["selection_count"]) != len(selected_rows):
            schema_errors.append(
                "selection_count in handoff_manifest.json does not match selected_maps.csv"
            )
    except Exception:
        schema_errors.append("selection_count in handoff_manifest.json is not an integer")

actual_selected_sha = _sha256(selected_path)
manifest_selected_sha = str(manifest.get("selection_csv_sha256", "")).strip()
if manifest_selected_sha and manifest_selected_sha != actual_selected_sha:
    schema_errors.append("selection_csv_sha256 mismatch for selected_maps.csv")

mask_requirements = {
    str(row.get("shortName", "")).strip(): str(row.get("required_mask_filename", "")).strip()
    for row in mask_rows
}

seen_short_names = set()
for idx, row in enumerate(selected_rows):
    short_name = str(row.get("shortName", "")).strip()
    if not short_name:
        schema_errors.append(f"selected_maps.csv row {idx} has empty shortName")
        continue
    if short_name in seen_short_names:
        schema_errors.append(f"selected_maps.csv has duplicate shortName: {short_name}")
    seen_short_names.add(short_name)
    if short_name not in mask_requirements:
        schema_errors.append(f"mask_requirements.csv missing shortName: {short_name}")

if schema_errors:
    for err in schema_errors:
        print(f"[SCHEMA] {err}", file=sys.stderr)
    sys.exit(EXIT_SCHEMA)

manifest_policy_path = str(manifest.get("tile_exclusion_policy_path", "")).strip()
if manifest_policy_path:
    manifest_policy_resolved = _resolve_path(manifest_policy_path, repo_root=repo_root)
else:
    manifest_policy_resolved = cli_policy_path

excluded_from_policy = _load_excluded_tiles(manifest_policy_resolved)
excluded_from_manifest = [str(v).strip() for v in manifest.get("excluded_tiles", []) if str(v).strip()]
excluded_union = set(excluded_from_policy) | set(excluded_from_manifest)

violations = sorted(excluded_union.intersection(seen_short_names))
if violations:
    for item in violations:
        print(
            f"[POLICY] excluded tile present in selected_maps.csv: {item}",
            file=sys.stderr,
        )
    sys.exit(EXIT_POLICY)

missing_images: list[str] = []
missing_sidecars: list[str] = []

for row in selected_rows:
    short_name = str(row.get("shortName", "")).strip()
    image_path_raw = str(row.get("image_path", "")).strip()
    image_filename = str(row.get("image_filename", "")).strip()

    candidates: list[Path] = []
    if image_path_raw:
        image_path = Path(image_path_raw)
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append(repo_root / image_path)
    if image_filename:
        candidates.append(images_dir / image_filename)

    resolved_image = None
    seen_candidates = set()
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
    for short_name in missing_images:
        print(f"[DATA] image missing for shortName={short_name}", file=sys.stderr)
    sys.exit(EXIT_DATA)

if missing_sidecars:
    for sidecar_name in missing_sidecars:
        print(f"[DATA] missing PNG sidecar: {sidecar_name}", file=sys.stderr)
    sys.exit(EXIT_DATA)

print(
    json.dumps(
        {
            "status": "ok",
            "handoff_dir": str(handoff_dir),
            "selection_count": len(selected_rows),
            "excluded_tiles_checked": len(excluded_union),
        },
        ensure_ascii=True,
    )
)
sys.exit(0)
PY
    ;;

  --help|-h|help)
    usage
    ;;

  *)
    echo "Unknown subcommand: $subcommand" >&2
    usage
    exit 1
    ;;
esac
