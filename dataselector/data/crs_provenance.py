"""Helpers for explicit CRS provenance resolution and auditing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from dataselector.data.spatial_schema import normalize_spatial_schema

_EPSG_AUTHORITY_RE = re.compile(r'AUTHORITY\["EPSG","(\d+)"\]', re.IGNORECASE)
_EPSG_LITERAL_RE = re.compile(r"EPSG[:\s]+(\d+)", re.IGNORECASE)
_NULL_STRINGS = {"", "nan", "none", "null"}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _NULL_STRINGS:
        return None
    return text


def normalize_crs_identifier(value: Any) -> str | None:
    """Normalize explicit CRS text to a compact identifier when possible."""

    text = _clean_text(value)
    if text is None:
        return None

    try:
        from pyproj import CRS

        crs = CRS.from_user_input(text)
        authority = crs.to_authority()
        if authority is not None and len(authority) == 2:
            return f"{str(authority[0]).upper()}:{authority[1]}"
        epsg = crs.to_epsg()
        if epsg is not None:
            return f"EPSG:{int(epsg)}"
        normalized = crs.to_string()
        return normalized or text
    except Exception:
        pass

    match = _EPSG_AUTHORITY_RE.search(text) or _EPSG_LITERAL_RE.search(text)
    if match:
        return f"EPSG:{match.group(1)}"
    return text if text.upper().startswith("EPSG:") else None


def crs_is_projected(source_crs: str | None) -> bool | None:
    normalized = normalize_crs_identifier(source_crs)
    if normalized is None:
        return None
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(normalized).is_projected)
    except Exception:
        if normalized in {"EPSG:4326", "EPSG:4258"}:
            return False
        if normalized.startswith("EPSG:"):
            return True
    return None


def find_aux_xml(image_path: Path, aux_dir: Path | None = None) -> Path | None:
    """Find a GDAL/PNG sidecar XML for an image path."""

    candidates = [
        image_path.with_suffix(image_path.suffix + ".aux.xml"),
        image_path.with_suffix(".aux.xml"),
    ]
    if aux_dir is not None:
        candidates.extend(
            [
                aux_dir / (image_path.name + ".aux.xml"),
                aux_dir / (image_path.stem + ".aux.xml"),
                aux_dir / (image_path.stem + image_path.suffix + ".aux.xml"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_crs_from_aux_xml(aux_path: Path) -> str | None:
    """Read explicit CRS information from a sidecar XML file."""

    try:
        root = ET.parse(aux_path).getroot()
    except Exception:
        return None

    srs_node = root.find(".//SRS")
    if srs_node is not None and srs_node.text:
        return normalize_crs_identifier(srs_node.text)
    return None


def extract_crs_from_raster(image_path: Path) -> str | None:
    """Read explicit CRS information from the raster dataset when available."""

    try:
        import rasterio
    except Exception:
        return None

    try:
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                return None
            return normalize_crs_identifier(dataset.crs.to_string())
    except Exception:
        return None


def resolve_explicit_crs_for_image(image_path: str | Path | None) -> dict[str, Any]:
    """Resolve explicit CRS provenance from sidecar first, then raster metadata."""

    raw = _clean_text(image_path)
    if raw is None:
        return {
            "source_crs": None,
            "crs_source": None,
            "crs_provenance": None,
            "aux_xml_path": None,
            "explicit": False,
        }

    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    aux_path = find_aux_xml(path)
    if aux_path is not None:
        source_crs = extract_crs_from_aux_xml(aux_path)
        if source_crs is not None:
            return {
                "source_crs": source_crs,
                "crs_source": "sidecar_xml",
                "crs_provenance": "explicit_sidecar_xml",
                "aux_xml_path": str(aux_path),
                "explicit": True,
            }

    source_crs = extract_crs_from_raster(path)
    if source_crs is not None:
        return {
            "source_crs": source_crs,
            "crs_source": "raster_dataset",
            "crs_provenance": "explicit_raster_dataset",
            "aux_xml_path": str(aux_path) if aux_path is not None else None,
            "explicit": True,
        }

    return {
        "source_crs": None,
        "crs_source": None,
        "crs_provenance": None,
        "aux_xml_path": str(aux_path) if aux_path is not None else None,
        "explicit": False,
    }


def audit_crs_provenance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit resolved CRS provenance and explicit/heuristic consistency."""

    work = df.copy()
    if "center_x" not in work.columns or "center_y" not in work.columns:
        work = normalize_spatial_schema(work, require_bounds=True, copy=False)

    rows: list[dict[str, Any]] = []
    explicit_values: set[str] = set()
    issues: list[str] = []

    for idx, row in work.iterrows():
        resolved_source = normalize_crs_identifier(row.get("source_crs"))
        declared_source = normalize_crs_identifier(row.get("source_crs_declared"))
        explicit = bool(row.get("crs_explicit"))
        crs_source = _clean_text(row.get("crs_source"))
        crs_provenance = _clean_text(row.get("crs_provenance"))
        short_name = _clean_text(row.get("shortName"))
        image_path = _clean_text(row.get("image_path"))
        identifier = short_name or image_path or str(idx)

        center_x = float(row["center_x"])
        center_y = float(row["center_y"])
        coordinates_projected = bool(abs(center_x) > 180.0 or abs(center_y) > 90.0)
        source_projected = crs_is_projected(resolved_source)

        status = "missing_explicit"
        note = None
        if explicit and resolved_source:
            explicit_values.add(resolved_source)
            if declared_source and declared_source != resolved_source:
                status = "declared_source_crs_mismatch"
                note = f"declared={declared_source}, resolved={resolved_source}"
            elif (
                source_projected is not None
                and source_projected != coordinates_projected
            ):
                status = "coordinate_unit_mismatch"
                coord_mode = "projected" if coordinates_projected else "geographic"
                source_mode = "projected" if source_projected else "geographic"
                note = f"coordinates={coord_mode}, source_crs={source_mode}"
            elif declared_source:
                status = "consistent_declared"
            else:
                status = "consistent_runtime_resolved"
        elif resolved_source:
            status = "heuristic_fallback"
            note = f"resolved via {crs_source or 'heuristic'}"

        if status in {"declared_source_crs_mismatch", "coordinate_unit_mismatch"}:
            issues.append(identifier)

        rows.append(
            {
                "row_index": int(idx),
                "shortName": short_name,
                "image_path": image_path,
                "source_crs_declared": declared_source,
                "source_crs": resolved_source,
                "crs_source": crs_source,
                "crs_provenance": crs_provenance,
                "crs_explicit": bool(explicit),
                "coordinates_projected": bool(coordinates_projected),
                "status": status,
                "note": note,
            }
        )

    audit_df = pd.DataFrame(rows)
    explicit_count = (
        int(audit_df["crs_explicit"].fillna(False).sum()) if not audit_df.empty else 0
    )
    heuristic_count = (
        int((audit_df["status"] == "heuristic_fallback").sum())
        if not audit_df.empty
        else 0
    )
    missing_count = (
        int((audit_df["status"] == "missing_explicit").sum())
        if not audit_df.empty
        else 0
    )
    mismatch_count = (
        int(
            audit_df["status"]
            .isin({"declared_source_crs_mismatch", "coordinate_unit_mismatch"})
            .sum()
        )
        if not audit_df.empty
        else 0
    )
    unique_explicit = sorted(explicit_values)

    if mismatch_count > 0:
        status = "consistency_mismatch"
    elif missing_count > 0:
        status = "explicit_missing"
    elif heuristic_count > 0:
        status = "heuristic_fallback_present"
    elif len(unique_explicit) > 1:
        status = "multiple_explicit_source_crs"
    elif explicit_count == len(audit_df):
        status = "explicit_uniform"
    else:
        status = "mixed"

    summary = {
        "crs_provenance_status": status,
        "crs_explicit_tile_count": explicit_count,
        "crs_heuristic_fallback_count": heuristic_count,
        "crs_missing_explicit_count": missing_count,
        "crs_consistency_issue_count": mismatch_count,
        "crs_consistency_issue_shortnames": issues,
        "crs_unique_explicit_source_crs": unique_explicit,
        "crs_explicit_source_crs": (
            unique_explicit[0] if len(unique_explicit) == 1 else None
        ),
        "crs_strict_ready": bool(
            mismatch_count == 0
            and missing_count == 0
            and heuristic_count == 0
            and len(unique_explicit) == 1
            and explicit_count == len(audit_df)
        ),
    }
    return audit_df, summary
