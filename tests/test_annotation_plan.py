from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataselector.runtime.parameter_snapshot import compute_file_sha256
from dataselector.workflows import annotation_plan as mod


def _write_dummy_aux_xml(path: Path, *, origin_x: float, origin_y: float) -> None:
    path.write_text(
        (
            "<PAMDataset>\n"
            f"  <GeoTransform>{origin_x:.6f}, 1.000000, 0.000000, {origin_y:.6f}, 0.000000, -1.000000</GeoTransform>\n"
            "</PAMDataset>\n"
        ),
        encoding="utf-8",
    )


def _write_dummy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "runs" / "dummy_run"
    image_dir = tmp_path / "data" / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)

    core_rows: list[dict[str, object]] = []
    for idx in range(28):
        short_name = f"KDR_{idx+1:03d}"
        city = f"City{idx+1:03d}"
        year = 1890 + (idx % 35)
        image_path = image_dir / f"{short_name}.png"
        arr = rng.integers(0, 255, size=(320, 320, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(image_path)
        _write_dummy_aux_xml(
            image_path.with_name(image_path.name + ".aux.xml"),
            origin_x=1000.0 + float(idx * 10),
            origin_y=2000.0 + float(idx * 10),
        )
        core_rows.append(
            {
                "shortName": short_name,
                "city": city,
                "year": year,
                "selection_rank": idx,
                "image_path": str(image_path),
            }
        )

    case_short_name = "KDR_146"
    case_img = image_dir / f"{case_short_name}.png"
    case_arr = rng.integers(0, 255, size=(320, 320, 3), dtype=np.uint8)
    Image.fromarray(case_arr, mode="RGB").save(case_img)
    _write_dummy_aux_xml(
        case_img.with_name(case_img.name + ".aux.xml"),
        origin_x=9000.0,
        origin_y=8000.0,
    )

    case_rows = [
        {
            "shortName": case_short_name,
            "city": "Hamburg",
            "year": 1918,
            "selection_rank": 0,
            "image_path": str(case_img),
        }
    ]

    final_rows = core_rows + [
        {
            "shortName": case_short_name,
            "city": "Hamburg",
            "year": 1918,
            "selection_rank": 28,
            "image_path": str(case_img),
        }
    ]

    pd.DataFrame(core_rows).to_csv(run_dir / "selection_core.csv", index=False)
    pd.DataFrame(case_rows).to_csv(run_dir / "selection_case.csv", index=False)
    pd.DataFrame(final_rows).to_csv(
        run_dir / "selection_final_with_cases.csv", index=False
    )

    (run_dir / "selection_contract.json").write_text(
        json.dumps(
            {
                "contract_version": "core_case_v1",
                "core_count": 28,
                "case_count_resolved": 1,
                "case_count_attached": 1,
                "final_count": 29,
                "case_tile_names": ["Hamburg"],
                "case_exclude_from_core": True,
                "case_attach_mode": "append_unique",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "core_count": 28,
                    "final_count": 29,
                }
            }
        ),
        encoding="utf-8",
    )

    return run_dir


def test_annotation_plan_generates_expected_manifest(tmp_path: Path):
    run_dir = _write_dummy_run(tmp_path)

    summary = mod.run_thesis_build_annotation_plan(
        run_dir=run_dir,
        patch_size=128,
        patches_per_tile=2,
        include_case=True,
        qc_mode="none",
        output_subdir="annotation_plan_test",
        split_n_splits=5,
    )

    assert summary["patches_total"] == 58
    assert summary["patches_qc_passed"] == 58
    assert summary["patches_qc_rejected"] == 0

    out_dir = Path(summary["output_dir"])
    manifest_df = pd.read_csv(out_dir / "patch_manifest.csv")
    assert len(manifest_df) == 58
    assert manifest_df["tile_shortname"].nunique() == 29
    assert (manifest_df.groupby("tile_shortname").size() == 2).all()
    assert "quicklook_aux_path" in manifest_df.columns
    assert "quicklook_has_georef" in manifest_df.columns
    assert manifest_df["quicklook_has_georef"].astype(bool).all()

    for row in manifest_df.itertuples(index=False):
        quicklook_rel = Path(str(row.quicklook_path))
        quicklook_aux_rel = Path(str(row.quicklook_aux_path))
        assert (out_dir / quicklook_rel).exists()
        assert (out_dir / quicklook_aux_rel).exists()

    contract = json.loads((out_dir / "annotation_dataset_contract.json").read_text("utf-8"))
    assert contract["source_hashes"]["selection_core"] == compute_file_sha256(
        run_dir / "selection_core.csv"
    )
    assert contract["source_hashes"]["selection_case"] == compute_file_sha256(
        run_dir / "selection_case.csv"
    )

    split_manifest = json.loads((out_dir / "patch_split_manifest.json").read_text("utf-8"))
    assert split_manifest["n_splits"] == 5
    assert isinstance(split_manifest["split_manifest_sha256"], str)
    assert len(split_manifest["split_manifest_sha256"]) == 64


def test_annotation_plan_deterministic_and_bounds(tmp_path: Path):
    run_dir = _write_dummy_run(tmp_path)

    out_a = mod.run_thesis_build_annotation_plan(
        run_dir=run_dir,
        patch_size=128,
        patches_per_tile=2,
        include_case=True,
        qc_mode="none",
        output_subdir="plan_a",
        split_n_splits=5,
    )
    out_b = mod.run_thesis_build_annotation_plan(
        run_dir=run_dir,
        patch_size=128,
        patches_per_tile=2,
        include_case=True,
        qc_mode="none",
        output_subdir="plan_b",
        split_n_splits=5,
    )

    a_df = pd.read_csv(Path(out_a["output_dir"]) / "patch_manifest.csv")
    b_df = pd.read_csv(Path(out_b["output_dir"]) / "patch_manifest.csv")

    cols = [
        "patch_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "selected_anchor_x",
        "selected_anchor_y",
        "split_fold",
    ]
    a_cmp = a_df[cols].sort_values("patch_id").reset_index(drop=True)
    b_cmp = b_df[cols].sort_values("patch_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a_cmp, b_cmp)

    assert (a_df["x0"] >= 0).all()
    assert (a_df["y0"] >= 0).all()
    assert (a_df["x1"] <= a_df["tile_width_px"]).all()
    assert (a_df["y1"] <= a_df["tile_height_px"]).all()
    assert ((a_df["x1"] - a_df["x0"]) == 128).all()
    assert ((a_df["y1"] - a_df["y0"]) == 128).all()

    # Group leakage guard: each tile in exactly one fold.
    passed = a_df[a_df["qc_status"] == "qc_passed"]
    fold_counts = passed.groupby("tile_shortname")["split_fold"].nunique()
    assert (fold_counts == 1).all()


def test_annotation_plan_fallback_records_replacement_reason(tmp_path: Path, monkeypatch):
    run_dir = _write_dummy_run(tmp_path)

    call_counter = {"count": 0}

    def fake_qc(_patch_img, *, qc_mode: str):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return mod.PatchQCDecision(
                passed=False,
                reason="forced_fail",
                metrics={"std_gray": 0.0},
            )
        return mod.PatchQCDecision(
            passed=True,
            reason=None,
            metrics={"std_gray": 10.0},
        )

    monkeypatch.setattr(mod, "_evaluate_patch_qc", fake_qc)

    out = mod.run_thesis_build_annotation_plan(
        run_dir=run_dir,
        patch_size=128,
        patches_per_tile=2,
        include_case=True,
        qc_mode="heuristic_v1",
        output_subdir="plan_fallback",
        split_n_splits=5,
    )

    manifest_df = pd.read_csv(Path(out["output_dir"]) / "patch_manifest.csv")
    first_row = manifest_df.sort_values(["selection_rank", "patch_index"]).iloc[0]

    assert bool(first_row["fallback_used"]) is True
    assert str(first_row["replacement_reason"]) == "forced_fail"
    assert str(first_row["qc_status"]) == "qc_passed"


def test_cli_smoke_build_annotation_plan(tmp_path: Path):
    run_dir = _write_dummy_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataselector",
            "thesis-build-annotation-plan",
            "--run-dir",
            str(run_dir),
            "--patch-size",
            "128",
            "--patches-per-tile",
            "2",
            "--qc-mode",
            "none",
            "--output-subdir",
            "annotation_plan_cli",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr

    out_dir = run_dir / "annotation_plan_cli"
    assert (out_dir / "patch_manifest.csv").exists()
    assert (out_dir / "patch_manifest.json").exists()
    assert (out_dir / "patch_qc_report.csv").exists()
    assert (out_dir / "patch_split_manifest.json").exists()
    assert (out_dir / "annotation_dataset_contract.json").exists()
    assert (out_dir / "class_mapping.yaml").exists()
    assert (out_dir / "annotation_progress.csv").exists()
    assert (out_dir / "annotation_qa_log.csv").exists()
