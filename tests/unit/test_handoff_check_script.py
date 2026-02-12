import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


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
""".strip()
        + "\n",
        encoding="utf-8",
    )

    images_dir = repo_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for short_name in ("KDR_101", "KDR_102"):
        (images_dir / f"{short_name}.png").write_bytes(b"PNG")
        (images_dir / f"{short_name}.png.aux.xml").write_text("<PAMDataset/>", encoding="utf-8")

    return repo_root, run_dir, source_selection_csv


def test_prepare_resolves_selection_from_report_and_writes_manifest_hash(tmp_path: Path) -> None:
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

    with (handoff_dir / "selected_maps.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["shortName"] == "KDR_201"


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
""".strip()
        + "\n",
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

    assert result_verify.returncode == 0, result_verify.stderr
