from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dataselector.workflows.handoff_bundle import (
    HandoffCheckError,
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


def _build_patch_bundle_fixture(
    repo_root: Path,
    *,
    run_name: str = "bundle_patch_filtered",
) -> dict[str, Path]:
    run_dir = repo_root / "outputs" / "runs" / run_name
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

    patch_specs = [
        ("KDR_201_p1", "KDR_201", 0),
        ("KDR_202_p1", "KDR_202", 1),
        ("KDR_203_p1", "KDR_203", 2),
    ]
    patch_rows: list[dict[str, object]] = []
    patch_to_fold: dict[str, int] = {}
    folds: list[dict[str, object]] = []
    for selection_rank, (patch_id, tile_shortname, split_fold) in enumerate(
        patch_specs
    ):
        (images_dir / f"{tile_shortname}.png").write_bytes(b"PNG")
        (images_dir / f"{tile_shortname}.png.aux.xml").write_text(
            "<PAMDataset/>",
            encoding="utf-8",
        )
        quicklook_rel = Path("quicklooks") / f"{patch_id}.tif"
        _write_dummy_quicklook_geotiff(annotation_dir / quicklook_rel)
        patch_rows.append(
            {
                "patch_id": patch_id,
                "tile_shortname": tile_shortname,
                "image_path": f"data/images/{tile_shortname}.png",
                "image_filename": f"{tile_shortname}.png",
                "x0": 0,
                "y0": 0,
                "x1": 32,
                "y1": 32,
                "qc_status": "qc_passed",
                "quicklook_path": str(quicklook_rel),
                "selection_rank": selection_rank,
                "selection_group": "core",
                "patch_index": 1,
                "patch_size_px": 32,
                "split_fold": split_fold,
            }
        )
        patch_to_fold[patch_id] = split_fold
        folds.append(
            {
                "fold": split_fold,
                "n_patches": 1,
                "n_tiles": 1,
                "patch_ids": [patch_id],
                "tile_shortnames": [tile_shortname],
            }
        )

    _write_csv(
        annotation_dir / "patch_manifest.csv",
        patch_rows,
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
        json.dumps(
            {
                "version": 1,
                "run_id": run_name,
                "grouping_key": "tile_shortname",
                "splitter": "GroupKFold",
                "n_splits": 3,
                "counts": {
                    "total_patches": 3,
                    "qc_passed_patches": 3,
                    "qc_rejected_patches": 0,
                    "unique_tiles": 3,
                },
                "folds": folds,
                "patch_to_fold": patch_to_fold,
            }
        ),
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
    return {
        "run_dir": run_dir,
        "policy_path": policy_path,
        "annotation_dir": annotation_dir,
    }


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


def test_handoff_bundle_patch_api_roundtrip_with_patch_id_filter(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    fixture = _build_patch_bundle_fixture(repo_root)
    filter_path = repo_root / "config" / "patch_filters" / "phase5_subset.txt"
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    filter_path.write_text(
        "# explicit subset for downstream annotation\nKDR_202_p1\nKDR_201_p1\nKDR_202_p1\n",
        encoding="utf-8",
    )

    out_dir = repo_root / "handoff" / "bundle_patch_filtered"
    prepare = prepare_patch_handoff(
        run_dir=fixture["run_dir"],
        out_dir=out_dir,
        patch_id_file=filter_path,
        repo_root=repo_root,
        tile_exclusion_policy=fixture["policy_path"],
    )
    verify = verify_patch_handoff(
        handoff_dir=out_dir,
        repo_root=repo_root,
        tile_exclusion_policy=fixture["policy_path"],
    )

    assert prepare["selection_count"] == 2
    assert Path(prepare["patch_id_filter_path"]).exists()
    assert verify["status"] == "ok"

    with (out_dir / "selected_patches.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        selected_rows = list(csv.DictReader(handle))
    assert [row["patch_id"] for row in selected_rows] == ["KDR_201_p1", "KDR_202_p1"]

    with (out_dir / "patch_mask_requirements.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        mask_rows = list(csv.DictReader(handle))
    assert [row["patch_id"] for row in mask_rows] == ["KDR_201_p1", "KDR_202_p1"]

    split_manifest = json.loads(
        (out_dir / "patch_split_manifest.json").read_text(encoding="utf-8")
    )
    assert split_manifest["counts"]["total_patches"] == 2
    assert split_manifest["counts"]["qc_passed_patches"] == 2
    assert set(split_manifest["patch_to_fold"]) == {"KDR_201_p1", "KDR_202_p1"}
    fold_patch_ids = {
        patch_id
        for fold in split_manifest["folds"]
        for patch_id in fold.get("patch_ids", [])
    }
    assert fold_patch_ids == {"KDR_201_p1", "KDR_202_p1"}

    manifest = json.loads(
        (out_dir / "patch_handoff_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["patch_selection_count"] == 2
    assert manifest["patch_id_filter_path"] == "patch_id_filter.txt"
    assert manifest["patch_id_filter_count"] == 2
    assert manifest["patch_id_filter_mode"] == "explicit_subset"
    assert (
        manifest["source_patch_id_filter_path"]
        == "config/patch_filters/phase5_subset.txt"
    )
    copied_filter = out_dir / "patch_id_filter.txt"
    assert copied_filter.read_text(encoding="utf-8") == filter_path.read_text(
        encoding="utf-8"
    )
    assert manifest["patch_id_filter_sha256"] == _sha256(copied_filter)


def _prepare_filtered_patch_handoff_for_verify(repo_root: Path) -> tuple[Path, Path]:
    fixture = _build_patch_bundle_fixture(repo_root, run_name="bundle_patch_filtered")
    filter_path = repo_root / "config" / "patch_filters" / "phase5_subset.txt"
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    filter_path.write_text(
        "# explicit subset for downstream annotation\nKDR_202_p1\nKDR_201_p1\nKDR_202_p1\n",
        encoding="utf-8",
    )
    out_dir = repo_root / "handoff" / "bundle_patch_filtered"
    prepare_patch_handoff(
        run_dir=fixture["run_dir"],
        out_dir=out_dir,
        patch_id_file=filter_path,
        repo_root=repo_root,
        tile_exclusion_policy=fixture["policy_path"],
    )
    return out_dir, fixture["policy_path"]


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing_sha", "patch_id_filter_sha256 missing"),
        ("empty_sha", "patch_id_filter_sha256 missing"),
        ("mismatched_sha", "patch_id_filter_sha256 mismatch"),
        ("missing_count", "patch_id_filter_count missing"),
        (
            "non_integer_count",
            "patch_id_filter_count in patch_handoff_manifest.json is not an integer",
        ),
        ("count_mismatch", "patch_id_filter_count mismatch"),
    ],
)
def test_verify_patch_handoff_rejects_invalid_patch_id_filter_manifest(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    out_dir, policy_path = _prepare_filtered_patch_handoff_for_verify(tmp_path)
    manifest_path = out_dir / "patch_handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if mutation == "missing_sha":
        manifest.pop("patch_id_filter_sha256")
    elif mutation == "empty_sha":
        manifest["patch_id_filter_sha256"] = ""
    elif mutation == "mismatched_sha":
        manifest["patch_id_filter_sha256"] = "0" * 64
    elif mutation == "missing_count":
        manifest.pop("patch_id_filter_count")
    elif mutation == "non_integer_count":
        manifest["patch_id_filter_count"] = "two"
    elif mutation == "count_mismatch":
        manifest["patch_id_filter_count"] = 3
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(f"Unhandled mutation: {mutation}")

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HandoffCheckError) as exc_info:
        verify_patch_handoff(
            handoff_dir=out_dir,
            repo_root=tmp_path,
            tile_exclusion_policy=policy_path,
        )

    assert any(expected_message in message for message in exc_info.value.messages)


def test_verify_patch_handoff_rejects_empty_patch_id_filter_file(
    tmp_path: Path,
) -> None:
    out_dir, policy_path = _prepare_filtered_patch_handoff_for_verify(tmp_path)
    copied_filter = out_dir / "patch_id_filter.txt"
    copied_filter.write_text("# comments only\n\n", encoding="utf-8")

    manifest_path = out_dir / "patch_handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patch_id_filter_sha256"] = _sha256(copied_filter)
    manifest["patch_id_filter_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HandoffCheckError) as exc_info:
        verify_patch_handoff(
            handoff_dir=out_dir,
            repo_root=tmp_path,
            tile_exclusion_policy=policy_path,
        )

    assert any(
        "patch-id filter file is empty" in message
        for message in exc_info.value.messages
    )


def test_prepare_patch_handoff_rejects_unknown_patch_id_filter_entry(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    fixture = _build_patch_bundle_fixture(repo_root, run_name="bundle_patch_unknown")
    filter_path = repo_root / "config" / "patch_filters" / "unknown_patch.txt"
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    filter_path.write_text("KDR_999_p1\n", encoding="utf-8")

    with pytest.raises(
        HandoffCheckError,
        match="patch-id filter references unknown patch_id: KDR_999_p1",
    ):
        prepare_patch_handoff(
            run_dir=fixture["run_dir"],
            out_dir=repo_root / "handoff" / "bundle_patch_unknown",
            patch_id_file=filter_path,
            repo_root=repo_root,
            tile_exclusion_policy=fixture["policy_path"],
        )
