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
