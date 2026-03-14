import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "handoff_check.sh"

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


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
    )


def test_handoff_check_script_is_package_wrapper() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "dataselector.workflows.handoff_bundle" in text


def _write_dummy_quicklook_geotiff(
    path: Path, *, origin_x: float = 1000.0, origin_y: float = 2000.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    transform = from_origin(origin_x, origin_y, 1.0, 1.0)
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


def _create_basic_repo_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "test_run_001"
    tuning_dir = run_dir / "tuning_weights"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = [
        {
            "shortName": "KDR_101",
            "image_path": "data/images/KDR_101.png",
            "image_filename": "KDR_101.png",
            "selection_rank": "0",
            "year": "1911",
            "city": "DemoCity",
            "city_source": "test",
            "longName": "KDR_101_DemoCity_1911.png",
            "center_x": "100.0",
            "center_y": "200.0",
        },
        {
            "shortName": "KDR_102",
            "image_path": "data/images/KDR_102.png",
            "image_filename": "KDR_102.png",
            "selection_rank": "1",
            "year": "1912",
            "city": "DemoTown",
            "city_source": "test",
            "longName": "KDR_102_DemoTown_1912.png",
            "center_x": "110.0",
            "center_y": "220.0",
        },
    ]
    source_selection_csv = run_dir / "selected_from_report.csv"
    _write_csv(
        source_selection_csv,
        selection_rows,
        fieldnames=[
            "shortName",
            "image_path",
            "image_filename",
            "selection_rank",
            "year",
            "city",
            "city_source",
            "longName",
            "center_x",
            "center_y",
        ],
    )

    report_text = "\n".join(
        [
            "# Thesis Pipeline Summary Report",
            "",
            "## Tile Selection",
            "- Selection file: `selected_from_report.csv`",
        ]
    )
    (run_dir / "THESIS_PIPELINE_REPORT.md").write_text(report_text, encoding="utf-8")

    snapshot_path = run_dir / "final_config_snapshot.yaml"
    snapshot_path.write_text("selection:\n  n_samples: 2\n", encoding="utf-8")
    snapshot_sha = _sha256(snapshot_path)

    run_metadata = {
        "extra": {
            "resolved_snapshot_path": str(snapshot_path),
            "resolved_snapshot_sha256": snapshot_sha,
        }
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        """
version: 1
rules:
  - id: exclude_kdr_155b_duplicate_variant
    action: exclude_from_candidate_pool
    match:
      shortName:
        - KDR_155b
""".strip() + "\n",
        encoding="utf-8",
    )

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for short_name in ("KDR_101", "KDR_102"):
        (images_dir / f"{short_name}.png").write_bytes(b"PNG")
        (images_dir / f"{short_name}.png.aux.xml").write_text(
            "<PAMDataset/>", encoding="utf-8"
        )

    return repo_root, run_dir, source_selection_csv


def test_prepare_resolves_selection_from_report_and_writes_manifest_hash(
    tmp_path: Path,
) -> None:
    repo_root, run_dir, _ = _create_basic_repo_fixture(tmp_path)
    handoff_dir = repo_root / "handoff" / "sel_report"

    result = _run_script(
        [
            "prepare",
            "--run-dir",
            str(run_dir),
            "--out",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )

    assert result.returncode == 0, result.stderr

    selected_maps = handoff_dir / "selected_maps.csv"
    manifest_path = handoff_dir / "handoff_manifest.json"
    mask_requirements = handoff_dir / "mask_requirements.csv"

    assert selected_maps.exists()
    assert manifest_path.exists()
    assert mask_requirements.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in manifest

    assert manifest["split_authority"] == "masterarbeit_strassenerkennung_cv"
    assert manifest["selection_csv_sha256"] == _sha256(selected_maps)


def test_prepare_falls_back_to_tuning_meta_when_report_missing(tmp_path: Path) -> None:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "test_run_002"
    tuning_dir = run_dir / "tuning_weights"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    alpha = 0.1
    beta = 0.2
    gamma = 0.7

    selection_csv = tuning_dir / f"selection_a{alpha}_b{beta}_g{gamma}.csv"
    _write_csv(
        selection_csv,
        [
            {
                "shortName": "KDR_201",
                "image_path": "data/images/KDR_201.png",
                "image_filename": "KDR_201.png",
                "selection_rank": "0",
            }
        ],
        fieldnames=["shortName", "image_path", "image_filename", "selection_rank"],
    )

    (run_dir / "tuning_weights" / "meta.json").write_text(
        json.dumps(
            {
                "best_metrics": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                }
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"extra": {}}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("version: 1\nrules: []\n", encoding="utf-8")

    handoff_dir = repo_root / "handoff" / "sel_fallback"
    result = _run_script(
        [
            "prepare",
            "--run-dir",
            str(run_dir),
            "--out",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )

    assert result.returncode == 0, result.stderr

    with (handoff_dir / "selected_maps.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["shortName"] == "KDR_201"


def test_prepare_prefers_selection_contract_core_for_thesis_v2_runs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "test_run_003"
    tuning_dir = run_dir / "tuning_weights"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    contract_core_csv = run_dir / "selection_core.csv"
    _write_csv(
        contract_core_csv,
        [
            {
                "shortName": "KDR_301",
                "image_path": "data/images/KDR_301.png",
                "image_filename": "KDR_301.png",
                "selection_rank": "0",
            }
        ],
        fieldnames=["shortName", "image_path", "image_filename", "selection_rank"],
    )

    snapshot_csv = run_dir / "selection_snapshot_primary.csv"
    _write_csv(
        snapshot_csv,
        [
            {
                "shortName": "KDR_301",
                "image_path": "data/images/KDR_301.png",
                "image_filename": "KDR_301.png",
                "selection_rank": "0",
            }
        ],
        fieldnames=["shortName", "image_path", "image_filename", "selection_rank"],
    )

    tuning_selection_csv = tuning_dir / "selection_a0.1_b0.2_g0.7.csv"
    _write_csv(
        tuning_selection_csv,
        [
            {
                "shortName": "KDR_999",
                "image_path": "data/images/KDR_999.png",
                "image_filename": "KDR_999.png",
                "selection_rank": "0",
            }
        ],
        fieldnames=["shortName", "image_path", "image_filename", "selection_rank"],
    )

    (tuning_dir / "meta.json").write_text(
        json.dumps(
            {
                "best_metrics": {
                    "alpha": 0.1,
                    "beta": 0.2,
                    "gamma": 0.7,
                }
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "selection_contract.json").write_text(
        json.dumps(
            {
                "selection_source": "snapshot_primary_selection",
                "selection_source_file": "selection_snapshot_primary.csv",
                "core_count": 1,
                "case_count": 0,
                "final_count": 1,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"extra": {}}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("version: 1\nrules: []\n", encoding="utf-8")

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for short_name in ("KDR_301", "KDR_999"):
        (images_dir / f"{short_name}.png").write_bytes(b"PNG")
        (images_dir / f"{short_name}.png.aux.xml").write_text(
            "<PAMDataset/>", encoding="utf-8"
        )

    handoff_dir = repo_root / "handoff" / "sel_contract"
    result = _run_script(
        [
            "prepare",
            "--run-dir",
            str(run_dir),
            "--out",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )

    assert result.returncode == 0, result.stderr

    manifest = json.loads(
        (handoff_dir / "handoff_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_selection_resolution"] == "selection_contract_core"

    with (handoff_dir / "selected_maps.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["shortName"] == "KDR_301"


def test_verify_local_detects_exclusion_policy_violation(tmp_path: Path) -> None:
    repo_root = tmp_path
    handoff_dir = repo_root / "handoff" / "sel_policy_violation"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        """
version: 1
rules:
  - id: exclude_kdr_155b_duplicate_variant
    action: exclude_from_candidate_pool
    match:
      shortName:
        - KDR_155b
""".strip() + "\n",
        encoding="utf-8",
    )

    _write_csv(
        handoff_dir / "selected_maps.csv",
        [
            {
                "shortName": "KDR_155b",
                "image_path": "data/images/KDR_155b.png",
                "image_filename": "KDR_155b.png",
                "selection_rank": "0",
            }
        ],
        fieldnames=["shortName", "image_path", "image_filename", "selection_rank"],
    )

    _write_csv(
        handoff_dir / "mask_requirements.csv",
        [{"shortName": "KDR_155b", "required_mask_filename": "KDR_155b_mask.tif"}],
        fieldnames=["shortName", "required_mask_filename"],
    )

    selected_sha = _sha256(handoff_dir / "selected_maps.csv")
    manifest = {
        "schema_version": "handoff_format_v1",
        "selection_id": "sel_violation",
        "run_id": "run_violation",
        "run_dir": "outputs/runs/run_violation",
        "resolved_snapshot_path": "outputs/runs/run_violation/final_config.yaml",
        "resolved_snapshot_sha256": "c" * 64,
        "selection_csv_path": "selected_maps.csv",
        "selection_csv_sha256": selected_sha,
        "selection_count": 1,
        "tile_exclusion_policy_path": "config/tile_exclusion_policy.yaml",
        "tile_exclusion_policy_sha256": _sha256(policy_path),
        "excluded_tiles": ["KDR_155b"],
        "split_authority": "masterarbeit_strassenerkennung_cv",
    }
    (handoff_dir / "handoff_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    result = _run_script(
        [
            "verify-local",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )

    assert result.returncode == 4
    assert "excluded tile present" in result.stderr


def test_verify_local_smoke_against_latest_thesis_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runs_dir = repo_root / "outputs" / "runs"
    images_dir = repo_root / "data" / "images"

    if not runs_dir.exists():
        pytest.skip("outputs/runs not available")
    if not images_dir.exists() or not list(images_dir.glob("*")):
        pytest.skip("data/images not available for local smoke")

    run_candidates = sorted(
        [path.parent for path in runs_dir.glob("*/THESIS_PIPELINE_REPORT.md")],
        reverse=True,
    )
    if not run_candidates:
        pytest.skip("No thesis runs with THESIS_PIPELINE_REPORT.md found")

    handoff_dir = tmp_path / "handoff_smoke"
    prepare_ok = False
    last_prepare = None

    for run_dir in run_candidates:
        result_prepare = _run_script(
            [
                "prepare",
                "--run-dir",
                str(run_dir),
                "--out",
                str(handoff_dir),
                "--repo-root",
                str(repo_root),
            ]
        )
        if result_prepare.returncode == 0:
            prepare_ok = True
            last_prepare = result_prepare
            break
        last_prepare = result_prepare

    if not prepare_ok:
        message = last_prepare.stderr if last_prepare else "unknown"
        pytest.skip(f"Could not prepare handoff from available runs: {message}")

    result_verify = _run_script(
        [
            "verify-local",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )

    if result_verify.returncode == 3:
        pytest.skip("Local raw image data/sidecars incomplete for smoke verification")
    if result_verify.returncode == 4:
        pytest.skip("Local smoke candidate violated current exclusion policy")

    assert result_verify.returncode == 0, result_verify.stderr


def _create_patch_run_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "runs" / "test_patch_run_001"
    annotation_dir = run_dir / "annotation_plan"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for short_name in ("KDR_301", "KDR_302"):
        (images_dir / f"{short_name}.png").write_bytes(b"PNG")
        (images_dir / f"{short_name}.png.aux.xml").write_text(
            "<PAMDataset><GeoTransform>1000,1,0,2000,0,-1</GeoTransform></PAMDataset>",
            encoding="utf-8",
        )

    quicklook_dir = annotation_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    for patch_name in ("KDR_301_p1", "KDR_302_p1", "KDR_302_p2"):
        _write_dummy_quicklook_geotiff(quicklook_dir / f"{patch_name}.tif")

    patch_rows = [
        {
            "patch_id": "KDR_301_p1",
            "tile_shortname": "KDR_301",
            "image_path": "data/images/KDR_301.png",
            "image_filename": "KDR_301.png",
            "x0": "0",
            "y0": "0",
            "x1": "1024",
            "y1": "1024",
            "split_fold": "1",
            "selection_rank": "0",
            "selection_group": "core",
            "patch_index": "1",
            "patch_size_px": "1024",
            "quicklook_path": "quicklooks/KDR_301_p1.tif",
            "qc_status": "qc_passed",
            "qc_reason": "",
        },
        {
            "patch_id": "KDR_302_p1",
            "tile_shortname": "KDR_302",
            "image_path": "data/images/KDR_302.png",
            "image_filename": "KDR_302.png",
            "x0": "64",
            "y0": "64",
            "x1": "1088",
            "y1": "1088",
            "split_fold": "2",
            "selection_rank": "1",
            "selection_group": "core",
            "patch_index": "1",
            "patch_size_px": "1024",
            "quicklook_path": "quicklooks/KDR_302_p1.tif",
            "qc_status": "qc_passed",
            "qc_reason": "",
        },
        {
            "patch_id": "KDR_302_p2",
            "tile_shortname": "KDR_302",
            "image_path": "data/images/KDR_302.png",
            "image_filename": "KDR_302.png",
            "x0": "256",
            "y0": "256",
            "x1": "1280",
            "y1": "1280",
            "split_fold": "2",
            "selection_rank": "1",
            "selection_group": "core",
            "patch_index": "2",
            "patch_size_px": "1024",
            "quicklook_path": "quicklooks/KDR_302_p2.tif",
            "qc_status": "qc_rejected",
            "qc_reason": "legend_dominant_edge",
        },
    ]
    _write_csv(
        annotation_dir / "patch_manifest.csv",
        patch_rows,
        fieldnames=[
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
        ],
    )

    (annotation_dir / "patch_split_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "n_splits": 5,
                "patch_to_fold": {
                    "KDR_301_p1": 1,
                    "KDR_302_p1": 2,
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    snapshot_path = run_dir / "final_config_snapshot.yaml"
    snapshot_path.write_text("selection:\n  n_samples: 2\n", encoding="utf-8")
    snapshot_sha = _sha256(snapshot_path)
    run_metadata = {
        "extra": {
            "resolved_snapshot_path": str(snapshot_path),
            "resolved_snapshot_sha256": snapshot_sha,
            "tile_excluded_shortnames": [],
            "tile_exclusion_policy_sha256": "",
        }
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("version: 1\nrules: []\n", encoding="utf-8")

    return repo_root, run_dir


def test_prepare_patches_and_verify_patches_roundtrip(tmp_path: Path) -> None:
    repo_root, run_dir = _create_patch_run_fixture(tmp_path)
    handoff_dir = repo_root / "handoff" / "patch_roundtrip"

    result_prepare = _run_script(
        [
            "prepare-patches",
            "--run-dir",
            str(run_dir),
            "--out",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result_prepare.returncode == 0, result_prepare.stderr

    selected_patches = handoff_dir / "selected_patches.csv"
    patch_manifest = handoff_dir / "patch_handoff_manifest.json"
    patch_masks = handoff_dir / "patch_mask_requirements.csv"
    split_manifest = handoff_dir / "patch_split_manifest.json"
    quicklook_p1 = handoff_dir / "quicklooks" / "KDR_301_p1.tif"
    assert selected_patches.exists()
    assert patch_manifest.exists()
    assert patch_masks.exists()
    assert split_manifest.exists()
    assert quicklook_p1.exists()

    with selected_patches.open("r", encoding="utf-8", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))
    assert len(selected_rows) == 2
    assert {row["patch_id"] for row in selected_rows} == {"KDR_301_p1", "KDR_302_p1"}

    manifest_payload = json.loads(patch_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["schema_version"] == "handoff_patch_format_v2"
    assert manifest_payload["patch_quicklook_format"] == "geotiff_deflate_rgb"
    assert manifest_payload["patch_selection_count"] == 2
    assert manifest_payload["patch_selection_csv_sha256"] == _sha256(selected_patches)

    result_verify = _run_script(
        [
            "verify-patches",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result_verify.returncode == 0, result_verify.stderr


def test_prepare_patches_rejects_legacy_png_quicklooks(tmp_path: Path) -> None:
    repo_root, run_dir = _create_patch_run_fixture(tmp_path)
    annotation_dir = run_dir / "annotation_plan"
    handoff_dir = repo_root / "handoff" / "patch_roundtrip_legacy_png"

    with (annotation_dir / "patch_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    legacy_rows: list[dict[str, str]] = []
    for row in rows:
        legacy_row = dict(row)
        legacy_row["quicklook_path"] = str(legacy_row["quicklook_path"]).replace(
            ".tif", ".png"
        )
        legacy_rows.append(legacy_row)
        legacy_png = annotation_dir / legacy_row["quicklook_path"]
        legacy_png.parent.mkdir(parents=True, exist_ok=True)
        legacy_png.write_bytes(b"PNG")

    _write_csv(
        annotation_dir / "patch_manifest.csv",
        legacy_rows,
        fieldnames=[
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
        ],
    )

    result_prepare = _run_script(
        [
            "prepare-patches",
            "--run-dir",
            str(run_dir),
            "--out",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result_prepare.returncode == 2
    assert "quicklook_path must end with .tif" in result_prepare.stderr


def test_verify_patches_detects_exclusion_policy_violation(tmp_path: Path) -> None:
    repo_root = tmp_path
    handoff_dir = repo_root / "handoff" / "patch_policy_violation"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        """
version: 1
rules:
  - id: exclude_kdr_155b_duplicate_variant
    action: exclude_from_candidate_pool
    match:
      shortName:
        - KDR_155b
""".strip() + "\n",
        encoding="utf-8",
    )

    _write_csv(
        handoff_dir / "selected_patches.csv",
        [
            {
                "patch_id": "KDR_155b_p1",
                "tile_shortname": "KDR_155b",
                "image_path": "data/images/KDR_155b.png",
                "image_filename": "KDR_155b.png",
                "x0": "0",
                "y0": "0",
                "x1": "1024",
                "y1": "1024",
                "split_fold": "1",
                "quicklook_path": "quicklooks/KDR_155b_p1.tif",
            }
        ],
        fieldnames=[
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
        ],
    )
    _write_csv(
        handoff_dir / "patch_mask_requirements.csv",
        [{"patch_id": "KDR_155b_p1", "required_mask_filename": "KDR_155b_p1_mask.tif"}],
        fieldnames=["patch_id", "required_mask_filename"],
    )
    (handoff_dir / "patch_split_manifest.json").write_text(
        json.dumps(
            {"version": 1, "n_splits": 5, "patch_to_fold": {"KDR_155b_p1": 1}},
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_sha = _sha256(handoff_dir / "selected_patches.csv")
    mask_sha = _sha256(handoff_dir / "patch_mask_requirements.csv")
    split_sha = _sha256(handoff_dir / "patch_split_manifest.json")
    manifest = {
        "schema_version": "handoff_patch_format_v2",
        "selection_id": "patch_sel_violation",
        "run_id": "run_violation",
        "run_dir": "outputs/runs/run_violation",
        "resolved_snapshot_path": "outputs/runs/run_violation/final_config.yaml",
        "resolved_snapshot_sha256": "c" * 64,
        "patch_selection_csv_path": "selected_patches.csv",
        "patch_selection_csv_sha256": selected_sha,
        "patch_selection_count": 1,
        "patch_mask_requirements_path": "patch_mask_requirements.csv",
        "patch_mask_requirements_sha256": mask_sha,
        "patch_split_manifest_path": "patch_split_manifest.json",
        "patch_split_manifest_sha256": split_sha,
        "patch_quicklook_format": "geotiff_deflate_rgb",
        "tile_exclusion_policy_path": "config/tile_exclusion_policy.yaml",
        "tile_exclusion_policy_sha256": _sha256(policy_path),
        "excluded_tiles": ["KDR_155b"],
        "split_authority": "masterarbeit_strassenerkennung_cv",
    }
    (handoff_dir / "patch_handoff_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    result = _run_script(
        [
            "verify-patches",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result.returncode == 4
    assert "excluded tile present" in result.stderr


def test_verify_patches_rejects_png_quicklook(tmp_path: Path) -> None:
    repo_root = tmp_path
    handoff_dir = repo_root / "handoff" / "patch_png_quicklook"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "KDR_401.png").write_bytes(b"PNG")
    (images_dir / "KDR_401.png.aux.xml").write_text(
        "<PAMDataset><GeoTransform>1000,1,0,2000,0,-1</GeoTransform></PAMDataset>",
        encoding="utf-8",
    )

    quicklook_dir = handoff_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    (quicklook_dir / "KDR_401_p1.png").write_bytes(b"PNG")

    _write_csv(
        handoff_dir / "selected_patches.csv",
        [
            {
                "patch_id": "KDR_401_p1",
                "tile_shortname": "KDR_401",
                "image_path": "data/images/KDR_401.png",
                "image_filename": "KDR_401.png",
                "x0": "0",
                "y0": "0",
                "x1": "1024",
                "y1": "1024",
                "split_fold": "1",
                "quicklook_path": "quicklooks/KDR_401_p1.png",
            }
        ],
        fieldnames=[
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
        ],
    )
    _write_csv(
        handoff_dir / "patch_mask_requirements.csv",
        [{"patch_id": "KDR_401_p1", "required_mask_filename": "KDR_401_p1_mask.tif"}],
        fieldnames=["patch_id", "required_mask_filename"],
    )
    (handoff_dir / "patch_split_manifest.json").write_text(
        json.dumps(
            {"version": 1, "n_splits": 5, "patch_to_fold": {"KDR_401_p1": 1}},
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("version: 1\nrules: []\n", encoding="utf-8")

    selected_sha = _sha256(handoff_dir / "selected_patches.csv")
    mask_sha = _sha256(handoff_dir / "patch_mask_requirements.csv")
    split_sha = _sha256(handoff_dir / "patch_split_manifest.json")
    manifest = {
        "schema_version": "handoff_patch_format_v2",
        "selection_id": "patch_sel_png_quicklook",
        "run_id": "run_png_quicklook",
        "run_dir": "outputs/runs/run_png_quicklook",
        "resolved_snapshot_path": "outputs/runs/run_png_quicklook/final_config.yaml",
        "resolved_snapshot_sha256": "a" * 64,
        "patch_selection_csv_path": "selected_patches.csv",
        "patch_selection_csv_sha256": selected_sha,
        "patch_selection_count": 1,
        "patch_mask_requirements_path": "patch_mask_requirements.csv",
        "patch_mask_requirements_sha256": mask_sha,
        "patch_split_manifest_path": "patch_split_manifest.json",
        "patch_split_manifest_sha256": split_sha,
        "patch_quicklook_format": "geotiff_deflate_rgb",
        "tile_exclusion_policy_path": "config/tile_exclusion_policy.yaml",
        "tile_exclusion_policy_sha256": _sha256(policy_path),
        "excluded_tiles": [],
        "split_authority": "masterarbeit_strassenerkennung_cv",
    }
    (handoff_dir / "patch_handoff_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    result = _run_script(
        [
            "verify-patches",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result.returncode == 3
    assert "quicklook format invalid" in result.stderr


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform, gcps, or rpcs.*")
def test_verify_patches_rejects_tif_without_georef(tmp_path: Path) -> None:
    repo_root = tmp_path
    handoff_dir = repo_root / "handoff" / "patch_tif_missing_georef"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "KDR_402.png").write_bytes(b"PNG")
    (images_dir / "KDR_402.png.aux.xml").write_text(
        "<PAMDataset><GeoTransform>1000,1,0,2000,0,-1</GeoTransform></PAMDataset>",
        encoding="utf-8",
    )

    quicklook_dir = handoff_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    with rasterio.open(
        quicklook_dir / "KDR_402_p1.tif",
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(arr)

    _write_csv(
        handoff_dir / "selected_patches.csv",
        [
            {
                "patch_id": "KDR_402_p1",
                "tile_shortname": "KDR_402",
                "image_path": "data/images/KDR_402.png",
                "image_filename": "KDR_402.png",
                "x0": "0",
                "y0": "0",
                "x1": "1024",
                "y1": "1024",
                "split_fold": "1",
                "quicklook_path": "quicklooks/KDR_402_p1.tif",
            }
        ],
        fieldnames=[
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
        ],
    )
    _write_csv(
        handoff_dir / "patch_mask_requirements.csv",
        [{"patch_id": "KDR_402_p1", "required_mask_filename": "KDR_402_p1_mask.tif"}],
        fieldnames=["patch_id", "required_mask_filename"],
    )
    (handoff_dir / "patch_split_manifest.json").write_text(
        json.dumps(
            {"version": 1, "n_splits": 5, "patch_to_fold": {"KDR_402_p1": 1}},
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_path = repo_root / "config" / "tile_exclusion_policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("version: 1\nrules: []\n", encoding="utf-8")

    selected_sha = _sha256(handoff_dir / "selected_patches.csv")
    mask_sha = _sha256(handoff_dir / "patch_mask_requirements.csv")
    split_sha = _sha256(handoff_dir / "patch_split_manifest.json")
    manifest = {
        "schema_version": "handoff_patch_format_v2",
        "selection_id": "patch_sel_tif_missing_georef",
        "run_id": "run_tif_missing_georef",
        "run_dir": "outputs/runs/run_tif_missing_georef",
        "resolved_snapshot_path": "outputs/runs/run_tif_missing_georef/final_config.yaml",
        "resolved_snapshot_sha256": "b" * 64,
        "patch_selection_csv_path": "selected_patches.csv",
        "patch_selection_csv_sha256": selected_sha,
        "patch_selection_count": 1,
        "patch_mask_requirements_path": "patch_mask_requirements.csv",
        "patch_mask_requirements_sha256": mask_sha,
        "patch_split_manifest_path": "patch_split_manifest.json",
        "patch_split_manifest_sha256": split_sha,
        "patch_quicklook_format": "geotiff_deflate_rgb",
        "tile_exclusion_policy_path": "config/tile_exclusion_policy.yaml",
        "tile_exclusion_policy_sha256": _sha256(policy_path),
        "excluded_tiles": [],
        "split_authority": "masterarbeit_strassenerkennung_cv",
    }
    (handoff_dir / "patch_handoff_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    result = _run_script(
        [
            "verify-patches",
            "--handoff-dir",
            str(handoff_dir),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert result.returncode == 3
    assert "quicklook georeference invalid" in result.stderr
