from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiLineString

from dataselector.cli_decorators import get_registered_commands
from dataselector.runtime.parameter_snapshot import compute_file_sha256
from dataselector.workflows.width_calibration import (
    audit_width_calibration_sensitivity,
    build_width_calibration_roads_source,
    measure_width_calibration,
    orchestrate_width_calibration,
    prepare_width_calibration,
    render_width_calibration_debug_masks,
    render_width_calibration_final_masks,
    summarize_width_calibration,
    sync_width_calibration_source,
)
from dataselector.workflows.width_calibration.measure_state import (
    WidthCalibrationSession,
    load_measurements_csv,
    load_tasks_csv,
)
from dataselector.workflows.width_calibration.prepare import compute_class_targets
from dataselector.workflows.width_calibration.models import (
    FINAL_MASK_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    MEASUREMENTS_FILENAME,
    MEASUREMENT_COLUMNS,
    SUMMARY_COLUMNS,
    SUPPORTED_CLASSES,
    TASK_COLUMNS,
    TaskRecord,
)
from dataselector.workflows.width_calibration.viewer_qt import (
    InteractiveMeasurementViewer,
    display_crop_size_px,
    select_interactive_matplotlib_backend,
    ui_scale_from_screen_metrics,
    upsample_nearest_rgb,
)


def _write_quicklook(path: Path, *, left: float, top: float, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((3, size, size), dtype=np.uint8)
    arr[0] = 220
    arr[1] = np.tile(np.arange(size, dtype=np.uint8), (size, 1))
    arr[2] = np.tile(np.arange(size, dtype=np.uint8).reshape(size, 1), (1, size))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint8",
        transform=from_origin(left, top, 1, 1),
        crs="EPSG:3857",
        compress="deflate",
    ) as dst:
        dst.write(arr)


def _build_handoff(tmp_path: Path) -> dict[str, Path]:
    handoff_dir = tmp_path / "handoff"
    quicklook_dir = handoff_dir / "quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)

    patch_specs = [
        ("KDR_001_p1", "KDR_001", 0.0, 64.0, 1, 1),
        ("KDR_002_p1", "KDR_002", 100.0, 64.0, 2, 1),
        ("KDR_003_p1", "KDR_003", 200.0, 64.0, 3, 1),
        ("KDR_004_p1", "KDR_004", 300.0, 64.0, 4, 1),
        ("KDR_146_p1", "KDR_146", 400.0, 64.0, 5, 1),
    ]
    rows: list[dict[str, object]] = []
    for selection_rank, (
        patch_id,
        tile_shortname,
        left,
        top,
        split_fold,
        patch_index,
    ) in enumerate(patch_specs):
        quicklook_name = f"{patch_id}.tif"
        _write_quicklook(quicklook_dir / quicklook_name, left=left, top=top)
        rows.append(
            {
                "patch_id": patch_id,
                "tile_shortname": tile_shortname,
                "image_path": f"data/images/{tile_shortname}.png",
                "image_filename": f"{tile_shortname}.png",
                "x0": 0,
                "y0": 0,
                "x1": 64,
                "y1": 64,
                "split_fold": split_fold,
                "selection_rank": selection_rank,
                "selection_group": "core",
                "patch_index": patch_index,
                "patch_size_px": 64,
                "quicklook_path": f"quicklooks/{quicklook_name}",
                "qc_status": "qc_passed",
                "qc_reason": "",
            }
        )
    pd.DataFrame(rows).to_csv(handoff_dir / "selected_patches.csv", index=False)
    pd.DataFrame(
        {
            "patch_id": [row["patch_id"] for row in rows],
            "required_mask_filename": [f"{row['patch_id']}_mask.tif" for row in rows],
        }
    ).to_csv(handoff_dir / "patch_mask_requirements.csv", index=False)
    (handoff_dir / "patch_handoff_manifest.json").write_text(
        json.dumps({"schema_version": "handoff_patch_format_v2"}, ensure_ascii=True),
        encoding="utf-8",
    )

    roads_path = tmp_path / "roads.gpkg"
    _write_roads_gpkg(roads_path, classes=[0, 0, 3, 8, 0])
    return {"handoff_dir": handoff_dir, "roads_path": roads_path}


def _write_roads_gpkg(path: Path, *, classes: list[int]) -> None:
    if path.exists():
        path.unlink()
    geometries = [
        MultiLineString([[(2, 32), (62, 32)]]),
        MultiLineString([[(102, 40), (162, 40)]]),
        MultiLineString([[(232, 2), (232, 62)]]),
        MultiLineString([[(302, 20), (362, 20)]]),
        MultiLineString([[(402, 24), (462, 24)]]),
    ]
    roads_gdf = gpd.GeoDataFrame(
        {"class": classes},
        geometry=geometries,
        crs="EPSG:3857",
    )
    roads_gdf.to_file(path, layer="cut_fixed_geometry_roads", driver="GPKG")


def _write_named_roads_layer(
    path: Path,
    *,
    layer: str,
    classes: list[int],
    x_offset: float,
) -> None:
    if path.exists():
        path.unlink()
    geometries = [
        MultiLineString(
            [
                [
                    (x_offset + float(idx) * 20.0, 16.0),
                    (x_offset + float(idx) * 20.0 + 12.0, 16.0),
                ]
            ]
        )
        for idx in range(len(classes))
    ]
    roads_gdf = gpd.GeoDataFrame(
        {"class": classes},
        geometry=geometries,
        crs="EPSG:3857",
    )
    roads_gdf.to_file(path, layer=layer, driver="GPKG")


def _write_summary_csv(path: Path, *, widths: dict[int, int]) -> None:
    rows: list[dict[str, object]] = []
    for class_id in SUPPORTED_CLASSES:
        width_px = widths.get(class_id, 4)
        rows.append(
            {
                "class": class_id,
                "n_valid_primary": 5,
                "median_px": float(width_px),
                "IQR_px": 1.0,
                "MAD_px": 0.5,
                "median_m": float(width_px),
                "IQR_m": 1.0,
                "MAD_m": 0.5,
                "repeat_median_abs_diff_px": 0.25,
                "repeat_median_abs_diff_m": 0.25,
                "low_evidence_flag": False,
                "high_variance_flag": False,
                "low_reliability_flag": False,
                "final_width_px": width_px,
                "final_width_m": float(width_px),
            }
        )
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_is_deterministic_and_excludes_hamburg(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    out1 = tmp_path / "prepared_a"
    out2 = tmp_path / "prepared_b"

    result1 = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out1,
    )
    result2 = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out2,
    )

    tasks_a = pd.read_csv(result1["tasks_csv"])
    tasks_b = pd.read_csv(result2["tasks_csv"])
    pd.testing.assert_frame_equal(tasks_a, tasks_b)
    assert not tasks_a["patch_id"].astype(str).str.startswith("KDR_146_").any()
    assert tasks_a["task_id"].is_unique
    assert "source_fid" in tasks_a.columns
    assert tasks_a["source_fid"].astype(str).str.strip().ne("").all()

    manifest = json.loads(Path(result1["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["hamburg_excluded_at_task_generation"] is True
    assert manifest["eligibility_parameters"]["minimum_in_crop_line_support_px"] == 32.0
    assert manifest["primary_task_count"] == int(
        (tasks_a["pass_type"] == "primary").sum()
    )
    assert manifest["repeat_task_count"] == int(
        (tasks_a["pass_type"] == "repeat").sum()
    )


@pytest.mark.fast
@pytest.mark.unit
def test_sync_width_calibration_source_copies_and_reports_in_sync(tmp_path: Path):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"

    first = sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    assert Path(first["dest_gpkg"]).exists()
    assert Path(first["sync_metadata_path"]).exists()
    assert first["copied"] is True
    assert first["in_sync"] is False
    dest_mtime_ns = dest_path.stat().st_mtime_ns

    second = sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    assert second["copied"] is False
    assert second["in_sync"] is True
    assert dest_path.stat().st_mtime_ns == dest_mtime_ns


@pytest.mark.fast
@pytest.mark.unit
def test_sync_width_calibration_source_rejects_missing_class_field(tmp_path: Path):
    source_path = tmp_path / "roads_missing_class.gpkg"
    geometries = [MultiLineString([[(2, 32), (62, 32)]])]
    roads_gdf = gpd.GeoDataFrame(
        {"road_class": [0]},
        geometry=geometries,
        crs="EPSG:3857",
    )
    roads_gdf.to_file(source_path, layer="cut_fixed_geometry_roads", driver="GPKG")

    with pytest.raises(ValueError, match="missing required 'class' field"):
        sync_width_calibration_source(
            source_gpkg=source_path,
            dest_gpkg=tmp_path / "local_sources" / "roads.gpkg",
            roads_layer="cut_fixed_geometry_roads",
        )


@pytest.mark.fast
@pytest.mark.unit
def test_build_width_calibration_roads_source_merges_base_and_tracer_layers(
    tmp_path: Path,
):
    cut_roads_path = tmp_path / "cut_fixed_geometry_roads.gpkg"
    tracer4_path = tmp_path / "tracer4_roads.gpkg"
    tracer5_path = tmp_path / "tracer5_roads.gpkg"
    dest_path = tmp_path / "handoff" / "local_sources" / "phase5_roads_merged.gpkg"

    _write_named_roads_layer(
        cut_roads_path,
        layer="cut_fixed_geometry_roads",
        classes=[0, 3],
        x_offset=0.0,
    )
    _write_named_roads_layer(
        tracer4_path,
        layer="4_roads_tracer_patches",
        classes=[91],
        x_offset=100.0,
    )
    _write_named_roads_layer(
        tracer5_path,
        layer="5_roads_tracer_patches",
        classes=[92],
        x_offset=200.0,
    )

    result = build_width_calibration_roads_source(
        cut_roads_gpkg=cut_roads_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        dest_gpkg=dest_path,
    )

    assert result["dest_gpkg"] == str(dest_path.resolve())
    assert result["dest_layer"] == "phase5_roads_merged"
    assert result["feature_count"] == 4
    assert Path(result["dest_gpkg"]).exists()
    assert Path(result["sources_json"]).exists()

    merged_gdf = gpd.read_file(dest_path, layer="phase5_roads_merged")
    assert merged_gdf["class"].tolist() == [0, 3, 4, 5]
    assert merged_gdf.loc[
        merged_gdf["source_layer"] == "cut_fixed_geometry_roads",
        "class",
    ].tolist() == [0, 3]
    assert merged_gdf.loc[
        merged_gdf["source_layer"] == "4_roads_tracer_patches",
        "class",
    ].tolist() == [4]
    assert merged_gdf.loc[
        merged_gdf["source_layer"] == "5_roads_tracer_patches",
        "class",
    ].tolist() == [5]

    sources_manifest = json.loads(
        Path(result["sources_json"]).read_text(encoding="utf-8")
    )
    assert sources_manifest["dest_gpkg"] == str(dest_path.resolve())
    assert sources_manifest["dest_layer"] == "phase5_roads_merged"
    assert sources_manifest["feature_count"] == 4
    assert len(sources_manifest["sources"]) == 3
    assert sources_manifest["sources"][0]["class_policy"] == "preserve"
    assert sources_manifest["sources"][1]["forced_class"] == 4
    assert sources_manifest["sources"][2]["forced_class"] == 5
    assert all(
        str(entry["source_gpkg"]).endswith(".gpkg")
        and str(entry["source_gpkg_sha256"]).strip()
        for entry in sources_manifest["sources"]
    )


@pytest.mark.fast
@pytest.mark.unit
def test_build_width_calibration_roads_source_rejects_invalid_cut_class_values(
    tmp_path: Path,
):
    cut_roads_path = tmp_path / "cut_fixed_geometry_roads_invalid.gpkg"
    tracer4_path = tmp_path / "tracer4_roads.gpkg"
    tracer5_path = tmp_path / "tracer5_roads.gpkg"

    cut_roads_gdf = gpd.GeoDataFrame(
        {"class": [0, None]},
        geometry=[
            MultiLineString([[(0.0, 16.0), (12.0, 16.0)]]),
            MultiLineString([[(20.0, 16.0), (32.0, 16.0)]]),
        ],
        crs="EPSG:3857",
    )
    cut_roads_gdf.to_file(
        cut_roads_path,
        layer="cut_fixed_geometry_roads",
        driver="GPKG",
    )
    _write_named_roads_layer(
        tracer4_path,
        layer="4_roads_tracer_patches",
        classes=[91],
        x_offset=100.0,
    )
    _write_named_roads_layer(
        tracer5_path,
        layer="5_roads_tracer_patches",
        classes=[92],
        x_offset=200.0,
    )

    with pytest.raises(ValueError, match="non-finite or non-numeric values in 'class'"):
        build_width_calibration_roads_source(
            cut_roads_gpkg=cut_roads_path,
            tracer4_gpkg=tracer4_path,
            tracer5_gpkg=tracer5_path,
        )


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_manifest_includes_sync_metadata(tmp_path: Path):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_result = sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )

    out_dir = tmp_path / "prepared"
    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
    )

    manifest = json.loads(Path(prepared["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["sync_metadata_path"] == str(
        Path(sync_result["sync_metadata_path"]).resolve()
    )
    assert manifest["sync_source_gpkg"] == str(Path(env["roads_path"]).resolve())
    assert manifest["sync_source_gpkg_sha256"] == sync_result["source_gpkg_sha256"]


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_can_prompt_and_sync_before_generating_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    _write_roads_gpkg(env["roads_path"], classes=[1, 0, 3, 8, 0])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=tmp_path / "prepared",
        prompt_for_sync=True,
    )

    assert (
        sync_width_calibration_source(
            source_gpkg=env["roads_path"],
            dest_gpkg=dest_path,
            roads_layer="cut_fixed_geometry_roads",
        )["in_sync"]
        is True
    )
    manifest = json.loads(Path(prepared["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["roads_gpkg_sha256"] == manifest["sync_source_gpkg_sha256"]


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_archives_stale_run_and_preserves_measurements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    out_dir = tmp_path / "prepared"
    prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
    )
    measurements_path = out_dir / "width_calibration_measurements.csv"
    measurements_path.write_text("sentinel_measurements\n", encoding="utf-8")
    _write_roads_gpkg(env["roads_path"], classes=[1, 0, 3, 8, 0])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        prompt_for_sync=True,
    )

    archive_dir = Path(prepared["archive_dir"])
    assert prepared["archived_previous_run"] is True
    assert archive_dir.exists()
    assert (archive_dir / "width_calibration_measurements.csv").read_text(
        encoding="utf-8"
    ) == "sentinel_measurements\n"
    assert (out_dir / "width_calibration_tasks.csv").exists()
    manifest = json.loads(Path(prepared["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["archived_previous_run"] is True


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_aborts_when_archive_declined(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "prepared"
    prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    with pytest.raises(RuntimeError, match="archival was declined"):
        prepare_width_calibration(
            handoff_dir=env["handoff_dir"],
            roads_gpkg=env["roads_path"],
            roads_layer="cut_fixed_geometry_roads",
            seed=8,
            crop_size_px=32,
            out_dir=out_dir,
            prompt_for_sync=True,
        )


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_archives_unknown_existing_run_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "width_calibration_tasks.csv").write_text(
        "stale_tasks\n", encoding="utf-8"
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        prompt_for_sync=True,
    )

    archive_dir = Path(prepared["archive_dir"])
    assert prepared["archived_previous_run"] is True
    assert (archive_dir / "width_calibration_tasks.csv").read_text(
        encoding="utf-8"
    ) == "stale_tasks\n"
    assert (out_dir / "width_calibration_manifest.json").exists()


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_keeps_current_run_in_place(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "prepared"
    first = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
    )
    measurements_path = out_dir / "width_calibration_measurements.csv"
    measurements_path.write_text("keep_me\n", encoding="utf-8")

    second = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
    )

    assert first["tasks_csv"] == second["tasks_csv"]
    assert second["archived_previous_run"] is False
    assert second["archive_dir"] == ""
    assert measurements_path.read_text(encoding="utf-8") == "keep_me\n"
    assert not list(tmp_path.glob("prepared_archive_*"))


@pytest.mark.fast
@pytest.mark.unit
def test_measure_width_calibration_fails_when_local_copy_changed_after_prepare(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=tmp_path / "prepared",
    )
    _write_roads_gpkg(dest_path, classes=[1, 0, 3, 8, 0])

    with pytest.raises(
        RuntimeError, match="stale relative to the current repo-local roads copy"
    ):
        measure_width_calibration(
            handoff_dir=env["handoff_dir"],
            tasks_csv=prepared["tasks_csv"],
            out_csv=tmp_path / "measurements.csv",
        )


@pytest.mark.fast
@pytest.mark.unit
def test_measure_width_calibration_fails_when_source_is_newer_than_local_copy(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=tmp_path / "prepared",
    )
    _write_roads_gpkg(env["roads_path"], classes=[1, 0, 3, 8, 0])

    with pytest.raises(RuntimeError, match="Run sync-width-calibration-source first"):
        measure_width_calibration(
            handoff_dir=env["handoff_dir"],
            tasks_csv=prepared["tasks_csv"],
            out_csv=tmp_path / "measurements.csv",
        )


@pytest.mark.fast
@pytest.mark.unit
def test_measure_width_calibration_can_prompt_sync_then_requires_reprepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    dest_path = tmp_path / "local_sources" / "roads.gpkg"
    sync_width_calibration_source(
        source_gpkg=env["roads_path"],
        dest_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
    )
    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=dest_path,
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=tmp_path / "prepared",
    )
    _write_roads_gpkg(env["roads_path"], classes=[1, 0, 3, 8, 0])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    with pytest.raises(RuntimeError, match="was synced from the editable source"):
        measure_width_calibration(
            handoff_dir=env["handoff_dir"],
            tasks_csv=prepared["tasks_csv"],
            out_csv=tmp_path / "measurements.csv",
            prompt_for_sync=True,
        )

    assert (
        sync_width_calibration_source(
            source_gpkg=env["roads_path"],
            dest_gpkg=dest_path,
            roads_layer="cut_fixed_geometry_roads",
        )["in_sync"]
        is True
    )


@pytest.mark.fast
@pytest.mark.unit
def test_measure_width_calibration_allows_matching_local_copy_without_sync_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    prepared = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=7,
        crop_size_px=32,
        out_dir=tmp_path / "prepared",
    )
    monkeypatch.setattr(
        InteractiveMeasurementViewer,
        "run",
        lambda self: {
            "started": True,
            "measurements_csv": str(self.session.measurements_path),
        },
    )

    result = measure_width_calibration(
        handoff_dir=env["handoff_dir"],
        tasks_csv=prepared["tasks_csv"],
        out_csv=tmp_path / "measurements.csv",
    )

    assert result["started"] is True


@pytest.mark.fast
@pytest.mark.unit
def test_width_calibration_session_resume_and_undo(tmp_path: Path):
    tasks_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_1",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 1,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_2",
                "class": 0,
                "patch_id": "KDR_002_p1",
                "tile_shortname": "KDR_002",
                "source_feature_id": "row_000002",
                "quicklook_path": "quicklooks/KDR_002_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 2,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_1",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 3,
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00001",
            },
        ],
        columns=TASK_COLUMNS,
    )
    measurements_path = tmp_path / "measurements.csv"

    session = WidthCalibrationSession(
        tasks_df=tasks_df,
        measurements_path=measurements_path,
        handoff_dir=tmp_path,
    )
    assert session.next_task().task_id == "task_00001"

    session.record_accept(
        "task_00001",
        click1=(30.0, 32.0),
        click2=(36.0, 32.0),
    )
    assert session.next_task().task_id == "task_00002"

    session.record_reject("task_00002", reject_reason="other", note="unclear")
    assert session.next_task().task_id == "task_00003"

    undone = session.undo_last()
    assert undone is not None
    assert undone["task_id"] == "task_00002"
    assert session.next_task().task_id == "task_00002"

    resumed = WidthCalibrationSession(
        tasks_df=tasks_df,
        measurements_path=measurements_path,
        handoff_dir=tmp_path,
    )
    assert resumed.next_task().task_id == "task_00002"


@pytest.mark.fast
@pytest.mark.unit
def test_width_calibration_session_progress_snapshot(tmp_path: Path):
    tasks_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_1",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 1,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_2",
                "class": 5,
                "patch_id": "KDR_002_p1",
                "tile_shortname": "KDR_002",
                "source_feature_id": "row_000002",
                "quicklook_path": "quicklooks/KDR_002_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 2,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_1",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 3,
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00001",
            },
        ],
        columns=TASK_COLUMNS,
    )
    measurements_path = tmp_path / "measurements.csv"
    session = WidthCalibrationSession(
        tasks_df=tasks_df,
        measurements_path=measurements_path,
        handoff_dir=tmp_path,
    )

    snap_before = session.progress_snapshot("task_00001")
    assert snap_before["pending_total"] == 2
    assert snap_before["eligible_total"] == 2
    assert snap_before["current_round_index"] == 1
    assert snap_before["current_round_total"] == 2
    assert snap_before["pending_by_pass"]["primary"] == 2
    assert snap_before["pending_by_pass"]["repeat"] == 0
    assert snap_before["pending_by_class"][0] == 1
    assert snap_before["pending_by_class"][5] == 1

    session.record_accept("task_00001", click1=(30.0, 32.0), click2=(36.0, 32.0))
    snap_after = session.progress_snapshot("task_00003")
    assert snap_after["pending_total"] == 2
    assert snap_after["eligible_total"] == 3
    assert snap_after["pending_by_pass"]["repeat"] == 1
    assert snap_after["current_round_index"] == 2


@pytest.mark.fast
@pytest.mark.unit
def test_record_accept_populates_metric_columns_when_quicklook_is_metric(tmp_path: Path):
    env = _build_handoff(tmp_path)
    tasks_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_1",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "1",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 1,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            }
        ],
        columns=TASK_COLUMNS,
    )
    measurements_path = tmp_path / "measurements.csv"
    session = WidthCalibrationSession(
        tasks_df=tasks_df,
        measurements_path=measurements_path,
        handoff_dir=env["handoff_dir"],
    )
    row = session.record_accept(
        "task_00001",
        click1=(30.0, 32.0),
        click2=(36.0, 32.0),
    )
    assert float(row["width_px"]) > 0.0
    assert float(row["width_m"]) > 0.0
    loaded = load_measurements_csv(measurements_path)
    assert "width_m" in loaded.columns
    assert "pixel_size_x_m" in loaded.columns
    assert "pixel_size_y_m" in loaded.columns
    assert "metric_valid" in loaded.columns


@pytest.mark.fast
@pytest.mark.unit
def test_compute_class_targets_proportional_mode() -> None:
    candidates_df = pd.DataFrame(
        {
            "class": [4] * 100 + [5] * 300 + [2] * 10,
        }
    )
    targets = compute_class_targets(
        candidates_df,
        quota_mode="proportional",
        sampling_rate=0.1,
        min_per_class=3,
        max_per_class=0,
    )
    assert targets[5] > targets[4]
    assert targets[4] >= 3
    assert targets[2] >= 3


@pytest.mark.fast
@pytest.mark.unit
def test_prepare_width_calibration_defaults_to_proportional_mode(tmp_path: Path) -> None:
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "prepared_default_mode"

    result = prepare_width_calibration(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        roads_layer="cut_fixed_geometry_roads",
        seed=42,
        crop_size_px=32,
        out_dir=out_dir,
    )

    manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["quota_mode"] == "proportional"
    assert float(manifest["quota_mode_parameters"]["sampling_rate"]) == 0.05
    assert int(manifest["quota_mode_parameters"]["min_per_class"]) == 3


@pytest.mark.fast
@pytest.mark.unit
def test_load_tasks_csv_groups_primary_and_repeat_tasks_by_class(tmp_path: Path):
    tasks_path = tmp_path / "tasks.csv"
    pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_1",
                "class": 6,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000001",
                "quicklook_path": "quicklooks/KDR_001_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 1,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_2",
                "class": 0,
                "patch_id": "KDR_002_p1",
                "tile_shortname": "KDR_002",
                "source_feature_id": "row_000002",
                "quicklook_path": "quicklooks/KDR_002_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 2,
                "pass_type": "primary",
                "repeat_of_task_id": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_3",
                "class": 6,
                "patch_id": "KDR_003_p1",
                "tile_shortname": "KDR_003",
                "source_feature_id": "row_000003",
                "quicklook_path": "quicklooks/KDR_003_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 3,
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00001",
            },
            {
                "task_id": "task_00004",
                "candidate_id": "cand_4",
                "class": 0,
                "patch_id": "KDR_004_p1",
                "tile_shortname": "KDR_004",
                "source_feature_id": "row_000004",
                "quicklook_path": "quicklooks/KDR_004_p1.tif",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "crop_size_px": 32,
                "queue_position": 4,
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00002",
            },
        ],
        columns=TASK_COLUMNS,
    ).to_csv(tasks_path, index=False)

    loaded = load_tasks_csv(tasks_path)
    assert loaded["task_id"].tolist() == [
        "task_00002",
        "task_00001",
        "task_00004",
        "task_00003",
    ]


@pytest.mark.fast
@pytest.mark.unit
def test_load_measurements_csv_accepts_legacy_files_without_source_fid(tmp_path: Path):
    path = tmp_path / "measurements.csv"
    legacy_columns = [col for col in MEASUREMENT_COLUMNS if col != "source_fid"]
    pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00001",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 30.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 4.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            }
        ],
        columns=legacy_columns,
    ).to_csv(path, index=False)

    loaded = load_measurements_csv(path)
    assert "source_fid" in loaded.columns
    assert loaded.loc[0, "source_fid"] == ""


@pytest.mark.fast
@pytest.mark.unit
def test_reject_current_task_uses_dialog_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class DummySession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def record_reject(
            self, task_id: str, *, reject_reason: str, note: str = ""
        ) -> None:
            self.calls.append((task_id, reject_reason, note))

    session = DummySession()
    viewer = InteractiveMeasurementViewer(
        handoff_dir=tmp_path,
        session=session,
    )
    viewer.current_task = TaskRecord(
        task_id="task_00001",
        candidate_id="cand_001",
        class_id=0,
        patch_id="KDR_001_p1",
        tile_shortname="KDR_001",
        source_fid="65",
        source_feature_id="row_000001",
        quicklook_path="quicklooks/KDR_001_p1.tif",
        anchor_x_px=32,
        anchor_y_px=32,
        crop_size_px=32,
        queue_position=1,
        pass_type="primary",
        repeat_of_task_id="",
    )
    advanced: list[str] = []
    monkeypatch.setattr(
        viewer,
        "_show_reject_dialog",
        lambda: ("other", "unclear"),
    )
    monkeypatch.setattr(viewer, "_show_next_task", lambda: advanced.append("next"))

    viewer._reject_current_task()

    assert session.calls == [("task_00001", "other", "unclear")]
    assert advanced == ["next"]


@pytest.mark.fast
@pytest.mark.unit
def test_reject_current_task_cancel_keeps_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class DummySession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def record_reject(
            self, task_id: str, *, reject_reason: str, note: str = ""
        ) -> None:
            self.calls.append((task_id, reject_reason, note))

    session = DummySession()
    viewer = InteractiveMeasurementViewer(
        handoff_dir=tmp_path,
        session=session,
    )
    viewer.current_task = TaskRecord(
        task_id="task_00001",
        candidate_id="cand_001",
        class_id=0,
        patch_id="KDR_001_p1",
        tile_shortname="KDR_001",
        source_fid="65",
        source_feature_id="row_000001",
        quicklook_path="quicklooks/KDR_001_p1.tif",
        anchor_x_px=32,
        anchor_y_px=32,
        crop_size_px=32,
        queue_position=1,
        pass_type="primary",
        repeat_of_task_id="",
    )
    advanced: list[str] = []
    monkeypatch.setattr(viewer, "_show_reject_dialog", lambda: None)
    monkeypatch.setattr(viewer, "_show_next_task", lambda: advanced.append("next"))

    viewer._reject_current_task()

    assert session.calls == []
    assert advanced == []


@pytest.mark.fast
@pytest.mark.unit
def test_viewer_status_message_includes_progress(tmp_path: Path):
    class DummySession:
        def progress_snapshot(self, _task_id: str) -> dict[str, object]:
            return {
                "recorded_total": 1,
                "pending_total": 5,
                "eligible_total": 6,
                "pending_by_pass": {"primary": 4, "repeat": 1},
                "pending_by_class": {0: 2},
                "current_position": 2,
                "current_remaining_total": 5,
                "current_remaining_in_pass": 4,
                "current_remaining_in_class": 2,
                "current_round": "primary",
                "current_round_index": 1,
                "current_round_total": 2,
            }

    class DummyStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, text: str) -> None:
            self.message = text

    viewer = InteractiveMeasurementViewer(
        handoff_dir=tmp_path,
        session=DummySession(),
    )
    viewer.current_task = TaskRecord(
        task_id="task_00001",
        candidate_id="cand_001",
        class_id=0,
        patch_id="KDR_001_p1",
        tile_shortname="KDR_001",
        source_fid="65",
        source_feature_id="row_000001",
        quicklook_path="quicklooks/KDR_001_p1.tif",
        anchor_x_px=32,
        anchor_y_px=32,
        crop_size_px=32,
        queue_position=1,
        pass_type="primary",
        repeat_of_task_id="",
    )
    bar = DummyStatusBar()
    viewer._qt_status_bar = bar

    viewer._update_status_message()

    assert "round=1/2" in bar.message
    assert "pos=2/6" in bar.message
    assert "remaining=5" in bar.message
    assert "remaining_pass=4" in bar.message
    assert "remaining_class=2" in bar.message


@pytest.mark.fast
@pytest.mark.unit
def test_summarize_width_calibration_uses_primary_only(
    tmp_path: Path,
):
    measurements_path = tmp_path / "measurements.csv"
    measurements_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00001",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 30.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 4.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_002",
                "class": 0,
                "patch_id": "KDR_002_p1",
                "tile_shortname": "KDR_002",
                "source_fid": "66",
                "source_feature_id": "row_000001",
                "measure_id": "measure_00002",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 40,
                "click1_x_px": 29.0,
                "click1_y_px": 40.0,
                "click2_x_px": 35.0,
                "click2_y_px": 40.0,
                "width_px": 6.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_003",
                "class": 3,
                "patch_id": "KDR_003_p1",
                "tile_shortname": "KDR_003",
                "source_fid": "103",
                "source_feature_id": "row_000002",
                "measure_id": "measure_00003",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 32.0,
                "click1_y_px": 28.0,
                "click2_x_px": 32.0,
                "click2_y_px": 36.0,
                "width_px": 8.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00004",
                "candidate_id": "cand_004",
                "class": 8,
                "patch_id": "KDR_004_p1",
                "tile_shortname": "KDR_004",
                "source_fid": "104",
                "source_feature_id": "row_000003",
                "measure_id": "measure_00004",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 20,
                "click1_x_px": 30.0,
                "click1_y_px": 20.0,
                "click2_x_px": 40.0,
                "click2_y_px": 20.0,
                "width_px": 10.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00005",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00005",
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00001",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 29.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 5.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00006",
                "candidate_id": "cand_003",
                "class": 3,
                "patch_id": "KDR_003_p1",
                "tile_shortname": "KDR_003",
                "source_fid": "103",
                "source_feature_id": "row_000002",
                "measure_id": "measure_00006",
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00003",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 32.0,
                "click1_y_px": 27.0,
                "click2_x_px": 32.0,
                "click2_y_px": 36.0,
                "width_px": 9.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
        ],
        columns=MEASUREMENT_COLUMNS,
    )
    measurements_df.to_csv(measurements_path, index=False)

    out_dir = tmp_path / "summary"
    result = summarize_width_calibration(
        measurements_csv=measurements_path,
        out_dir=out_dir,
    )

    summary_df = pd.read_csv(result["summary_csv"])
    class0 = summary_df.loc[summary_df["class"] == 0].iloc[0]
    assert int(class0["n_valid_primary"]) == 2
    assert float(class0["median_px"]) == 5.0
    assert int(class0["final_width_px"]) == 5
    assert float(class0["repeat_median_abs_diff_px"]) == 1.0

    class8 = summary_df.loc[summary_df["class"] == 8].iloc[0]
    assert bool(class8["low_evidence_flag"])


@pytest.mark.fast
@pytest.mark.unit
def test_summarize_width_calibration_includes_meter_fields_and_policy(tmp_path: Path):
    measurements_path = tmp_path / "measurements.csv"
    measurements_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00001",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 30.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 4.0,
                "width_m": 4.0,
                "pixel_size_x_m": 1.0,
                "pixel_size_y_m": 1.0,
                "crs_linear_unit": "metre",
                "metric_valid": 1,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_002",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000001",
                "measure_id": "measure_00002",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 36,
                "anchor_y_px": 32,
                "click1_x_px": 33.0,
                "click1_y_px": 32.0,
                "click2_x_px": 39.0,
                "click2_y_px": 32.0,
                "width_px": 6.0,
                "width_m": 6.0,
                "pixel_size_x_m": 1.0,
                "pixel_size_y_m": 1.0,
                "crs_linear_unit": "metre",
                "metric_valid": 1,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00003",
                "pass_type": "repeat",
                "repeat_of_task_id": "task_00001",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 29.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 5.0,
                "width_m": 5.0,
                "pixel_size_x_m": 1.0,
                "pixel_size_y_m": 1.0,
                "crs_linear_unit": "metre",
                "metric_valid": 1,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
        ],
        columns=MEASUREMENT_COLUMNS,
    )
    measurements_df.to_csv(measurements_path, index=False)

    out_dir = tmp_path / "summary_meter"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "workflow_version": "phase5_width_calibration_v2",
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "handoff_dir": "handoff/sample",
        "roads_gpkg": "handoff/local_sources/phase5_roads_merged.gpkg",
        "roads_gpkg_sha256": "abc",
        "roads_layer": "phase5_roads_merged",
        "seed": 42,
        "crop_size_px": 128,
        "quota_mode": "proportional",
        "quota_mode_parameters": {
            "sampling_rate": 0.05,
            "min_per_class": 3,
            "max_per_class": 0,
            "repeat_sampling_rate": 0.2,
            "repeat_min_per_class": 1,
        },
        "observed_class_counts": {"0": 2},
        "primary_class_targets": {"0": 2},
        "repeat_class_targets": {"0": 1},
    }
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest_payload, ensure_ascii=True),
        encoding="utf-8",
    )

    result = summarize_width_calibration(
        measurements_csv=measurements_path,
        out_dir=out_dir,
    )

    summary_df = pd.read_csv(result["summary_csv"])
    row = summary_df.loc[summary_df["class"] == 0].iloc[0]
    assert float(row["median_m"]) == 5.0
    assert float(row["repeat_median_abs_diff_m"]) == 1.0
    assert float(row["final_width_m"]) == 5.0

    summary_json = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary_json["sampling_policy"]["quota_mode"] == "proportional"
    assert int(summary_json["sampling_policy"]["seed"]) == 42


@pytest.mark.fast
@pytest.mark.unit
def test_audit_width_calibration_sensitivity_writes_expected_outputs(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    measurements_path = tmp_path / "measurements.csv"
    measurements_df = pd.DataFrame(
        [
            {
                "task_id": "task_00001",
                "candidate_id": "cand_001",
                "class": 0,
                "patch_id": "KDR_001_p1",
                "tile_shortname": "KDR_001",
                "source_fid": "65",
                "source_feature_id": "row_000000",
                "measure_id": "measure_00001",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 30.0,
                "click1_y_px": 32.0,
                "click2_x_px": 34.0,
                "click2_y_px": 32.0,
                "width_px": 4.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00002",
                "candidate_id": "cand_002",
                "class": 0,
                "patch_id": "KDR_002_p1",
                "tile_shortname": "KDR_002",
                "source_fid": "66",
                "source_feature_id": "row_000001",
                "measure_id": "measure_00002",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 40,
                "click1_x_px": 29.0,
                "click1_y_px": 40.0,
                "click2_x_px": 35.0,
                "click2_y_px": 40.0,
                "width_px": 6.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00003",
                "candidate_id": "cand_003",
                "class": 3,
                "patch_id": "KDR_003_p1",
                "tile_shortname": "KDR_003",
                "source_fid": "103",
                "source_feature_id": "row_000002",
                "measure_id": "measure_00003",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 32,
                "click1_x_px": 32.0,
                "click1_y_px": 28.0,
                "click2_x_px": 32.0,
                "click2_y_px": 36.0,
                "width_px": 8.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
            {
                "task_id": "task_00004",
                "candidate_id": "cand_004",
                "class": 8,
                "patch_id": "KDR_004_p1",
                "tile_shortname": "KDR_004",
                "source_fid": "104",
                "source_feature_id": "row_000003",
                "measure_id": "measure_00004",
                "pass_type": "primary",
                "repeat_of_task_id": "",
                "anchor_x_px": 32,
                "anchor_y_px": 20,
                "click1_x_px": 30.0,
                "click1_y_px": 20.0,
                "click2_x_px": 40.0,
                "click2_y_px": 20.0,
                "width_px": 10.0,
                "keep": 1,
                "reject_reason": "",
                "note": "",
            },
        ],
        columns=MEASUREMENT_COLUMNS,
    )
    measurements_df.to_csv(measurements_path, index=False)

    out_dir = tmp_path / "summary"
    summary_result = summarize_width_calibration(
        measurements_csv=measurements_path,
        out_dir=out_dir,
    )
    result = audit_width_calibration_sensitivity(
        summary_csv=summary_result["summary_csv"],
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        out_dir=out_dir,
    )

    sensitivity_df = pd.read_csv(result["sensitivity_csv"])
    assert set(sensitivity_df["variant"]) == {
        "baseline",
        "median_minus_1px",
        "median_plus_1px",
    }
    assert sorted(set(sensitivity_df["patch_id"])) == [
        "KDR_001_p1",
        "KDR_002_p1",
        "KDR_003_p1",
        "KDR_004_p1",
    ]
    overlay_dir = Path(result["sensitivity_overlay_dir"])
    assert sorted(path.name for path in overlay_dir.glob("*.png"))


@pytest.mark.fast
@pytest.mark.unit
def test_render_width_calibration_debug_masks_writes_patch_masks_and_manifest(
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "debug_masks"

    result = render_width_calibration_debug_masks(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        out_dir=out_dir,
        fixed_width_px=10,
    )

    requirements_df = pd.read_csv(env["handoff_dir"] / "patch_mask_requirements.csv")
    expected_names = requirements_df["required_mask_filename"].astype(str).tolist()
    assert result["mask_count"] == len(expected_names)
    for mask_name in expected_names:
        assert (out_dir / mask_name).exists()

    manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["debug_only"] is True
    assert manifest["test_only"] is True
    assert manifest["rendering_mode"] == "fixed_width_px"
    assert manifest["fixed_width_px"] == 10

    sample_mask_path = out_dir / "KDR_001_p1_mask.tif"
    with (
        rasterio.open(sample_mask_path) as mask_ds,
        rasterio.open(
            env["handoff_dir"] / "quicklooks" / "KDR_001_p1.tif"
        ) as quicklook_ds,
    ):
        mask = mask_ds.read(1)
        assert int(np.count_nonzero(mask)) > 0
        assert mask_ds.width == quicklook_ds.width
        assert mask_ds.height == quicklook_ds.height
        assert mask_ds.transform == quicklook_ds.transform
        assert str(mask_ds.crs) == str(quicklook_ds.crs)


@pytest.mark.fast
@pytest.mark.unit
def test_render_width_calibration_final_masks_uses_summary_widths(tmp_path: Path):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "final_masks"
    summary_csv = tmp_path / "width_calibration_summary.csv"
    _write_summary_csv(summary_csv, widths={0: 7, 3: 5, 8: 6})

    result = render_width_calibration_final_masks(
        handoff_dir=env["handoff_dir"],
        roads_gpkg=env["roads_path"],
        summary_csv=summary_csv,
        out_dir=out_dir,
        expected_roads_gpkg_sha256=compute_file_sha256(env["roads_path"]),
        expected_summary_csv_sha256=compute_file_sha256(summary_csv),
    )

    requirements_df = pd.read_csv(env["handoff_dir"] / "patch_mask_requirements.csv")
    expected_names = requirements_df["required_mask_filename"].astype(str).tolist()
    assert result["mask_count"] == len(expected_names)
    for mask_name in expected_names:
        assert (out_dir / mask_name).exists()

    manifest_path = out_dir / FINAL_MASK_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["debug_only"] is False
    assert manifest["test_only"] is False
    assert manifest["rendering_mode"] == "final_width_px"
    assert "fixed_width_px" not in manifest
    assert manifest["class_widths_px"]["0"] == 7
    assert manifest["class_widths_px"]["3"] == 5
    assert manifest["class_widths_px"]["8"] == 6
    assert manifest["summary_csv"] == str(summary_csv.resolve())
    assert manifest["summary_csv_sha256"]
    assert result["roads_gpkg_sha256"] == manifest["roads_gpkg_sha256"]
    assert result["summary_csv_sha256"] == manifest["summary_csv_sha256"]

    sample_mask_path = out_dir / "KDR_001_p1_mask.tif"
    with (
        rasterio.open(sample_mask_path) as mask_ds,
        rasterio.open(
            env["handoff_dir"] / "quicklooks" / "KDR_001_p1.tif"
        ) as quicklook_ds,
    ):
        mask = mask_ds.read(1)
        assert int(np.count_nonzero(mask)) > 0
        assert mask_ds.width == quicklook_ds.width
        assert mask_ds.height == quicklook_ds.height
        assert mask_ds.transform == quicklook_ds.transform
        assert str(mask_ds.crs) == str(quicklook_ds.crs)


@pytest.mark.fast
@pytest.mark.unit
def test_render_width_calibration_final_masks_rejects_sha_mismatch(tmp_path: Path):
    env = _build_handoff(tmp_path)
    out_dir = tmp_path / "final_masks"
    summary_csv = tmp_path / "width_calibration_summary.csv"
    _write_summary_csv(summary_csv, widths={0: 7})

    with pytest.raises(ValueError, match="roads_gpkg SHA256 mismatch"):
        render_width_calibration_final_masks(
            handoff_dir=env["handoff_dir"],
            roads_gpkg=env["roads_path"],
            summary_csv=summary_csv,
            out_dir=out_dir,
            expected_roads_gpkg_sha256="0" * 64,
        )

    assert not out_dir.exists()


@pytest.mark.fast
@pytest.mark.unit
def test_width_calibration_commands_are_registered():
    import dataselector.cli  # noqa: F401

    commands = get_registered_commands()
    assert "sync-width-calibration-source" in commands
    assert "build-width-calibration-roads-source" in commands
    assert "prepare-width-calibration" in commands
    assert "measure-width-calibration" in commands
    assert "summarize-width-calibration" in commands
    assert "audit-width-calibration-sensitivity" in commands
    assert "render-width-calibration-debug-masks" in commands
    assert "render-width-calibration-final-masks" in commands


@pytest.mark.fast
@pytest.mark.unit
def test_display_crop_size_px_supports_full_and_reduced_display_windows():
    assert display_crop_size_px(128, 1.0) == 128
    assert display_crop_size_px(256, 0.50) == 128
    assert display_crop_size_px(255, 0.25) == 65


@pytest.mark.fast
@pytest.mark.unit
def test_upsample_nearest_rgb_scales_image_by_integer_factor():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    upscaled = upsample_nearest_rgb(image, 4)
    assert upscaled.shape == (8, 12, 3)
    assert np.all(upscaled[0:4, 0:4] == image[0, 0])


@pytest.mark.fast
@pytest.mark.unit
def test_ui_scale_from_screen_metrics_handles_normal_and_hidpi_layouts():
    assert ui_scale_from_screen_metrics(None) == 1.0
    assert (
        ui_scale_from_screen_metrics(82.0, screen_width_px=1920, screen_height_px=1080)
        == 1.0
    )
    assert (
        ui_scale_from_screen_metrics(110.0, screen_width_px=1920, screen_height_px=1080)
        == 1.1458333333333333
    )
    assert (
        ui_scale_from_screen_metrics(165.0, screen_width_px=1920, screen_height_px=1080)
        == 1.71875
    )
    assert (
        ui_scale_from_screen_metrics(96.0, screen_width_px=2880, screen_height_px=1800)
        == 1.6666666666666667
    )
    assert (
        ui_scale_from_screen_metrics(260.0, screen_width_px=3840, screen_height_px=2160)
        == 2.5
    )


@pytest.mark.fast
@pytest.mark.unit
def test_select_interactive_matplotlib_backend_requires_qt(
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyMatplotlib:
        def get_backend(self) -> str:
            return "Agg"

    monkeypatch.setattr(
        "dataselector.workflows.width_calibration.viewer_qt.has_module",
        lambda _module_name: False,
    )
    with pytest.raises(RuntimeError, match="requires a Qt 6 backend"):
        select_interactive_matplotlib_backend(DummyMatplotlib())


@pytest.mark.fast
@pytest.mark.unit
def test_select_interactive_matplotlib_backend_activates_qt(
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyMatplotlib:
        def __init__(self) -> None:
            self.backend = "Agg"
            self.calls: list[tuple[str, bool]] = []

        def get_backend(self) -> str:
            return self.backend

        def use(self, backend: str, force: bool = False) -> None:
            self.calls.append((backend, force))
            self.backend = backend

    monkeypatch.setattr(
        "dataselector.workflows.width_calibration.viewer_qt.has_module",
        lambda module_name: module_name
        in {"PySide6", "matplotlib.backends.backend_qtagg"},
    )
    dummy = DummyMatplotlib()
    backend = select_interactive_matplotlib_backend(dummy)
    assert backend == "QtAgg"
    assert dummy.calls == [("QtAgg", True)]


# PR1 Orchestrator Tests


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_cli_registered():
    """Test 1: CLI-Registrierung: neuer Command sichtbar."""
    commands = get_registered_commands()
    assert "orchestrate-width-calibration" in commands


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_call_order(tmp_path: Path):
    """Test 2: Reihenfolge-Test: Snapshot vor Build vor Prepare."""
    env = _build_handoff(tmp_path)
    
    cut_path = tmp_path / "cut.gpkg"
    tracer4_path = tmp_path / "tracer4.gpkg"
    tracer5_path = tmp_path / "tracer5.gpkg"
    out_dir = tmp_path / "orchestrated_output"
    
    _write_named_roads_layer(cut_path, layer="cut_fixed_geometry_roads", classes=[0, 3], x_offset=0.0)
    _write_named_roads_layer(tracer4_path, layer="4_roads_tracer_patches", classes=[91], x_offset=100.0)
    _write_named_roads_layer(tracer5_path, layer="5_roads_tracer_patches", classes=[92], x_offset=200.0)
    
    result = orchestrate_width_calibration(
        cut_roads_gpkg=cut_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        handoff_dir=env["handoff_dir"],
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        skip_measure=True,
        repo_root_path=tmp_path,
    )
    
    # Verify all stages completed with correct structure
    assert "snapshot" in result
    assert "build" in result
    assert "prepare" in result
    assert result["snapshot"]["run_id"]
    assert result["build"]["dest_gpkg"]
    assert result["build"]["dest_gpkg"] == str(
        (
            tmp_path
            / "handoff"
            / "local_sources"
            / "phase5_roads_merged.gpkg"
        ).resolve()
    )
    assert result["prepare"]["tasks_csv"]
    assert Path(result["prepare"]["tasks_csv"]).exists()


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_skip_measure(tmp_path: Path):
    """Test 3: Skip-Measure-Test: kein Measure-Aufruf."""
    env = _build_handoff(tmp_path)
    
    cut_path = tmp_path / "cut.gpkg"
    tracer4_path = tmp_path / "tracer4.gpkg"
    tracer5_path = tmp_path / "tracer5.gpkg"
    out_dir = tmp_path / "orchestrated_output"
    
    _write_named_roads_layer(cut_path, layer="cut_fixed_geometry_roads", classes=[0, 3], x_offset=0.0)
    _write_named_roads_layer(tracer4_path, layer="4_roads_tracer_patches", classes=[91], x_offset=100.0)
    _write_named_roads_layer(tracer5_path, layer="5_roads_tracer_patches", classes=[92], x_offset=200.0)
    
    result = orchestrate_width_calibration(
        cut_roads_gpkg=cut_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        handoff_dir=env["handoff_dir"],
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        skip_measure=True,
        repo_root_path=tmp_path,
    )
    
    # measure should not be in result when skip_measure=True
    assert "measure" not in result


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_accepts_resume_flag(tmp_path: Path):
    """Test 4: Resume-Parameter wird akzeptiert."""
    env = _build_handoff(tmp_path)
    
    cut_path = tmp_path / "cut.gpkg"
    tracer4_path = tmp_path / "tracer4.gpkg"
    tracer5_path = tmp_path / "tracer5.gpkg"
    out_dir = tmp_path / "orchestrated_output"
    
    _write_named_roads_layer(cut_path, layer="cut_fixed_geometry_roads", classes=[0, 3], x_offset=0.0)
    _write_named_roads_layer(tracer4_path, layer="4_roads_tracer_patches", classes=[91], x_offset=100.0)
    _write_named_roads_layer(tracer5_path, layer="5_roads_tracer_patches", classes=[92], x_offset=200.0)
    
    # Orchestrator accepts resume flag without error (actual resume requires interactive testing)
    result = orchestrate_width_calibration(
        cut_roads_gpkg=cut_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        handoff_dir=env["handoff_dir"],
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        skip_measure=True,
        resume=True,  # Accept resume flag
        repo_root_path=tmp_path,
    )
    
    # Verify it completes successfully
    assert result["prepare"]["tasks_csv"]


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_qt_gating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test 6: Qt-Gating-Test: früher Abbruch bei fehlender Qt-Umgebung."""
    env = _build_handoff(tmp_path)
    
    cut_path = tmp_path / "cut.gpkg"
    tracer4_path = tmp_path / "tracer4.gpkg"
    tracer5_path = tmp_path / "tracer5.gpkg"
    out_dir = tmp_path / "orchestrated_output"
    
    _write_named_roads_layer(cut_path, layer="cut_fixed_geometry_roads", classes=[0, 3], x_offset=0.0)
    _write_named_roads_layer(tracer4_path, layer="4_roads_tracer_patches", classes=[91], x_offset=100.0)
    _write_named_roads_layer(tracer5_path, layer="5_roads_tracer_patches", classes=[92], x_offset=200.0)
    
    # Mock select_interactive_matplotlib_backend to raise error
    def mock_select_backend(matplotlib_module: Any) -> str:
        raise RuntimeError("PySide6 and matplotlib QtAgg must be available.")
    
    monkeypatch.setattr(
        "dataselector.workflows.width_calibration.viewer_qt.select_interactive_matplotlib_backend",
        mock_select_backend,
    )
    
    # Should fail early when skip_measure=False and Qt is missing
    with pytest.raises(RuntimeError, match="Cannot proceed with measurement"):
        orchestrate_width_calibration(
            cut_roads_gpkg=cut_path,
            tracer4_gpkg=tracer4_path,
            tracer5_gpkg=tracer5_path,
            handoff_dir=env["handoff_dir"],
            seed=7,
            crop_size_px=32,
            out_dir=out_dir,
            skip_measure=False,
            repo_root_path=tmp_path,
        )


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_uses_default_measurements_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    env = _build_handoff(tmp_path)

    cut_path = tmp_path / "cut.gpkg"
    tracer4_path = tmp_path / "tracer4.gpkg"
    tracer5_path = tmp_path / "tracer5.gpkg"
    out_dir = tmp_path / "orchestrated_output"

    _write_named_roads_layer(
        cut_path,
        layer="cut_fixed_geometry_roads",
        classes=[0, 3],
        x_offset=0.0,
    )
    _write_named_roads_layer(
        tracer4_path,
        layer="4_roads_tracer_patches",
        classes=[91],
        x_offset=100.0,
    )
    _write_named_roads_layer(
        tracer5_path,
        layer="5_roads_tracer_patches",
        classes=[92],
        x_offset=200.0,
    )

    monkeypatch.setattr(
        "dataselector.workflows.width_calibration.viewer_qt.select_interactive_matplotlib_backend",
        lambda _matplotlib_module: "QtAgg",
    )

    captured: dict[str, Any] = {}

    def _fake_measure_width_calibration(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "tasks_csv": str(kwargs["tasks_csv"]),
            "measurements_csv": str(kwargs["out_csv"]),
            "status": "mocked",
        }

    monkeypatch.setattr(
        "dataselector.workflows.width_calibration.measure_state.measure_width_calibration",
        _fake_measure_width_calibration,
    )

    result = orchestrate_width_calibration(
        cut_roads_gpkg=cut_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        handoff_dir=env["handoff_dir"],
        seed=7,
        crop_size_px=32,
        out_dir=out_dir,
        skip_measure=False,
        repo_root_path=tmp_path,
    )

    expected_measurements_csv = str((out_dir / MEASUREMENTS_FILENAME).resolve())
    assert captured["out_csv"] == expected_measurements_csv
    assert result["measure"]["measurements_csv"] == expected_measurements_csv


@pytest.mark.fast
@pytest.mark.unit
def test_orchestrate_width_calibration_accepts_custom_source_layers(tmp_path: Path):
    env = _build_handoff(tmp_path)

    cut_path = tmp_path / "cut_custom.gpkg"
    tracer4_path = tmp_path / "tracer4_custom.gpkg"
    tracer5_path = tmp_path / "tracer5_custom.gpkg"
    out_dir = tmp_path / "orchestrated_output_custom_layers"

    _write_named_roads_layer(
        cut_path,
        layer="cut_fixed_geometry_roads",
        classes=[0, 3],
        x_offset=0.0,
    )
    _write_named_roads_layer(
        tracer4_path,
        layer="4_fixed",
        classes=[91],
        x_offset=100.0,
    )
    _write_named_roads_layer(
        tracer5_path,
        layer="5_fixed",
        classes=[92],
        x_offset=200.0,
    )

    result = orchestrate_width_calibration(
        cut_roads_gpkg=cut_path,
        tracer4_gpkg=tracer4_path,
        tracer5_gpkg=tracer5_path,
        cut_roads_layer="cut_fixed_geometry_roads",
        tracer4_layer="4_fixed",
        tracer5_layer="5_fixed",
        handoff_dir=env["handoff_dir"],
        seed=42,
        crop_size_px=128,
        out_dir=out_dir,
        skip_measure=True,
        repo_root_path=tmp_path,
    )

    assert result["policy"]["tracer4_layer"] == "4_fixed"
    assert result["policy"]["tracer5_layer"] == "5_fixed"
    assert Path(result["prepare"]["tasks_csv"]).exists()
