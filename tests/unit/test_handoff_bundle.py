from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from dataselector.workflows.handoff_bundle import (
    prepare_patch_handoff,
    prepare_tile_handoff,
    verify_patch_handoff,
    verify_tile_handoff,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dummy_quicklook_geotiff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
        compress="DEFLATE",
        predictor=2,
    ) as dst:
        dst.write(arr)


def _write_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "version: 1\n"
            "rules:\n"
            "  - id: exclude_kdr_155b_duplicate_variant\n"
            "    action: exclude_from_candidate_pool\n"
            "    match:\n"
            "      shortName:\n"
            "        - KDR_155b\n"
        ),
        encoding="utf-8",
    )


def test_handoff_bundle_tile_api_roundtrip(tmp_path: Path) -> None:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "bundle_tile"
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_core = run_dir / "selection_core.csv"
    _write_csv(
        selection_core,
        [
            {
                "shortName": "KDR_101",
                "image_path": "data/images/KDR_101.png",
                "image_filename": "KDR_101.png",
                "selection_rank": 0,
                "year": 1911,
                "city": "DemoCity",
            }
        ],
        ["shortName", "image_path", "image_filename", "selection_rank", "year", "city"],
    )
    (run_dir / "selection_contract.json").write_text(
        json.dumps({"selection_source": "snapshot_primary_selection"}),
        encoding="utf-8",
    )
    snapshot_path = run_dir / "final_config.yaml"
    snapshot_path.write_text("parameters: {}\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "resolved_snapshot_path": str(snapshot_path),
                    "resolved_snapshot_sha256": _sha256(snapshot_path),
                }
            }
        ),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    _write_policy(policy_path)

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "KDR_101.png").write_bytes(b"PNG")
    (images_dir / "KDR_101.png.aux.xml").write_text("<PAMDataset/>", encoding="utf-8")

    out_dir = repo_root / "handoff" / "bundle_tile"
    prepare = prepare_tile_handoff(
        run_dir=run_dir,
        out_dir=out_dir,
        repo_root=repo_root,
        tile_exclusion_policy=policy_path,
    )
    verify = verify_tile_handoff(
        handoff_dir=out_dir,
        repo_root=repo_root,
        tile_exclusion_policy=policy_path,
    )

    assert prepare["selection_source"] == "selection_contract_core"
    assert prepare["selection_count"] == 1
    assert verify["status"] == "ok"
    manifest = json.loads(
        (out_dir / "handoff_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "handoff_format_v1"
    assert manifest["selection_count"] == 1


def test_handoff_bundle_patch_api_roundtrip(tmp_path: Path) -> None:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "bundle_patch"
    annotation_dir = run_dir / "annotation_plan"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "final_config.yaml"
    snapshot_path.write_text("parameters: {}\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "resolved_snapshot_path": str(snapshot_path),
                    "resolved_snapshot_sha256": _sha256(snapshot_path),
                }
            }
        ),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    _write_policy(policy_path)

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "KDR_201.png").write_bytes(b"PNG")
    (images_dir / "KDR_201.png.aux.xml").write_text("<PAMDataset/>", encoding="utf-8")

    quicklook_rel = Path("quicklooks") / "KDR_201_p1.tif"
    _write_dummy_quicklook_geotiff(annotation_dir / quicklook_rel)

    _write_csv(
        annotation_dir / "patch_manifest.csv",
        [
            {
                "patch_id": "KDR_201_p1",
                "tile_shortname": "KDR_201",
                "image_path": "data/images/KDR_201.png",
                "image_filename": "KDR_201.png",
                "x0": 0,
                "y0": 0,
                "x1": 32,
                "y1": 32,
                "qc_status": "qc_passed",
                "quicklook_path": str(quicklook_rel),
                "selection_rank": 0,
                "selection_group": "core",
                "patch_index": 1,
                "patch_size_px": 32,
                "split_fold": 0,
            }
        ],
        [
            "patch_id",
            "tile_shortname",
            "image_path",
            "image_filename",
            "x0",
            "y0",
            "x1",
            "y1",
            "qc_status",
            "quicklook_path",
            "selection_rank",
            "selection_group",
            "patch_index",
            "patch_size_px",
            "split_fold",
        ],
    )
    (annotation_dir / "patch_split_manifest.json").write_text(
        json.dumps({"patch_to_fold": {"KDR_201_p1": 0}}),
        encoding="utf-8",
    )
    (annotation_dir / "annotation_dataset_contract.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "patch_manifest_csv": "annotation_plan/patch_manifest.csv",
                    "patch_split_manifest_json": "annotation_plan/patch_split_manifest.json",
                }
            }
        ),
        encoding="utf-8",
    )

    out_dir = repo_root / "handoff" / "bundle_patch"
    prepare = prepare_patch_handoff(
        run_dir=run_dir,
        out_dir=out_dir,
        repo_root=repo_root,
        tile_exclusion_policy=policy_path,
    )
    verify = verify_patch_handoff(
        handoff_dir=out_dir,
        repo_root=repo_root,
        tile_exclusion_policy=policy_path,
    )

    assert prepare["selection_count"] == 1
    assert verify["status"] == "ok"
    manifest = json.loads(
        (out_dir / "patch_handoff_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "handoff_patch_format_v2"
    assert manifest["patch_selection_count"] == 1
