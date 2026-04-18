from __future__ import annotations

import json

from dataselector.cli_decorators import cli_command

from .audit import audit_width_calibration_sensitivity
from .measure_state import measure_width_calibration
from .models import DEFAULT_DISPLAY_CROP_FACTOR, DEFAULT_DISPLAY_SCALE
from .prepare import prepare_width_calibration
from .render import render_width_calibration_debug_masks
from .runs import (
    build_width_calibration_roads_source,
    orchestrate_width_calibration,
    sync_width_calibration_source,
)
from .summary import summarize_width_calibration


@cli_command(
    "sync-width-calibration-source",
    help="Sync the editable QGIS roads GeoPackage into the repo-local width-calibration source path",
    args={
        "source_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the editable source roads GeoPackage",
        },
        "dest_gpkg": {
            "type": str,
            "required": False,
            "default": None,
            "help": "Destination repo-local roads GeoPackage path; defaults to handoff/local_sources/cut_fixed_geometry_roads.gpkg",
        },
        "roads_layer": {
            "type": str,
            "required": False,
            "default": "cut_fixed_geometry_roads",
            "help": "Layer name inside the source roads GeoPackage",
        },
    },
)
def sync_width_calibration_source_cmd(
    source_gpkg: str,
    dest_gpkg: str | None = None,
    roads_layer: str = "cut_fixed_geometry_roads",
) -> int:
    print(
        json.dumps(
            sync_width_calibration_source(
                source_gpkg=source_gpkg,
                dest_gpkg=dest_gpkg,
                roads_layer=roads_layer,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "build-width-calibration-roads-source",
    help="Build the canonical repo-local Phase-5 roads GeoPackage from the classified base layer plus tracer patch layers",
    args={
        "cut_roads_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the classified cut roads GeoPackage",
        },
        "tracer4_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the tracer GeoPackage whose features map to class 4",
        },
        "tracer5_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the tracer GeoPackage whose features map to class 5",
        },
        "dest_gpkg": {
            "type": str,
            "required": False,
            "default": None,
            "help": "Destination repo-local merged roads GeoPackage path; defaults to handoff/local_sources/phase5_roads_merged.gpkg",
        },
        "cut_roads_layer": {
            "type": str,
            "required": False,
            "default": "cut_fixed_geometry_roads",
            "help": "Layer name inside the classified cut roads GeoPackage",
        },
        "tracer4_layer": {
            "type": str,
            "required": False,
            "default": "4_roads_tracer_patches",
            "help": "Layer name inside the class-4 tracer GeoPackage",
        },
        "tracer5_layer": {
            "type": str,
            "required": False,
            "default": "5_roads_tracer_patches",
            "help": "Layer name inside the class-5 tracer GeoPackage",
        },
        "dest_layer": {
            "type": str,
            "required": False,
            "default": "phase5_roads_merged",
            "help": "Layer name written into the merged roads GeoPackage",
        },
    },
)
def build_width_calibration_roads_source_cmd(
    cut_roads_gpkg: str,
    tracer4_gpkg: str,
    tracer5_gpkg: str,
    dest_gpkg: str | None = None,
    cut_roads_layer: str = "cut_fixed_geometry_roads",
    tracer4_layer: str = "4_roads_tracer_patches",
    tracer5_layer: str = "5_roads_tracer_patches",
    dest_layer: str = "phase5_roads_merged",
) -> int:
    print(
        json.dumps(
            build_width_calibration_roads_source(
                cut_roads_gpkg=cut_roads_gpkg,
                tracer4_gpkg=tracer4_gpkg,
                tracer5_gpkg=tracer5_gpkg,
                dest_gpkg=dest_gpkg,
                cut_roads_layer=cut_roads_layer,
                tracer4_layer=tracer4_layer,
                tracer5_layer=tracer5_layer,
                dest_layer=dest_layer,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "orchestrate-width-calibration",
    help="Orchestrate complete width-calibration workflow: Snapshot -> Build -> Prepare -> optional Measure",
    args={
        "cut_roads_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the classified cut roads GeoPackage",
        },
        "tracer4_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the tracer GeoPackage whose features map to class 4",
        },
        "tracer5_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the tracer GeoPackage whose features map to class 5",
        },
        "handoff_dir": {
            "type": str,
            "required": True,
            "help": "Path to the Phase-5 handoff directory",
        },
        "seed": {
            "type": int,
            "required": True,
            "help": "Random seed for deterministic task generation",
        },
        "crop_size_px": {
            "type": int,
            "required": True,
            "help": "Crop size shown during interactive measurement",
        },
        "out_dir": {
            "type": str,
            "required": True,
            "help": "Output directory for task and manifest artifacts",
        },
        "skip_measure": {
            "type": bool,
            "action": "store_true",
            "help": "Skip the interactive measurement stage; only snapshot, build, and prepare",
        },
        "resume": {
            "type": bool,
            "action": "store_true",
            "help": "Resume an existing measurement session instead of starting fresh",
        },
        "quota_mode": {
            "type": str,
            "required": False,
            "default": "fixed",
            "help": "Prepare sampling mode: fixed (legacy) or proportional",
        },
        "sampling_rate": {
            "type": float,
            "required": False,
            "default": 0.05,
            "help": "Per-class sampling rate used when quota_mode=proportional",
        },
        "min_per_class": {
            "type": int,
            "required": False,
            "default": 3,
            "help": "Minimum primary tasks per class for proportional mode",
        },
        "max_per_class": {
            "type": int,
            "required": False,
            "default": 0,
            "help": "Optional cap for primary tasks per class; 0 disables cap",
        },
        "repeat_sampling_rate": {
            "type": float,
            "required": False,
            "default": 0.2,
            "help": "Per-class repeat sampling rate used when quota_mode=proportional",
        },
        "repeat_min_per_class": {
            "type": int,
            "required": False,
            "default": 1,
            "help": "Minimum repeat tasks per class in proportional mode",
        },
        "display_crop_factor": {
            "type": float,
            "default": DEFAULT_DISPLAY_CROP_FACTOR,
            "help": "Fraction of the prepared crop shown around the anchor point",
        },
        "display_scale": {
            "type": int,
            "default": DEFAULT_DISPLAY_SCALE,
            "help": "Nearest-neighbor display scaling factor for the shown crop",
        },
    },
)
def orchestrate_width_calibration_cmd(
    cut_roads_gpkg: str,
    tracer4_gpkg: str,
    tracer5_gpkg: str,
    handoff_dir: str,
    seed: int,
    crop_size_px: int,
    out_dir: str,
    skip_measure: bool = False,
    resume: bool = False,
    quota_mode: str = "fixed",
    sampling_rate: float = 0.05,
    min_per_class: int = 3,
    max_per_class: int = 0,
    repeat_sampling_rate: float = 0.2,
    repeat_min_per_class: int = 1,
    display_crop_factor: float = DEFAULT_DISPLAY_CROP_FACTOR,
    display_scale: int = DEFAULT_DISPLAY_SCALE,
) -> int:
    print(
        json.dumps(
            orchestrate_width_calibration(
                cut_roads_gpkg=cut_roads_gpkg,
                tracer4_gpkg=tracer4_gpkg,
                tracer5_gpkg=tracer5_gpkg,
                handoff_dir=handoff_dir,
                seed=seed,
                crop_size_px=crop_size_px,
                out_dir=out_dir,
                skip_measure=bool(skip_measure),
                resume=bool(resume),
                quota_mode=quota_mode,
                sampling_rate=float(sampling_rate),
                min_per_class=int(min_per_class),
                max_per_class=int(max_per_class),
                repeat_sampling_rate=float(repeat_sampling_rate),
                repeat_min_per_class=int(repeat_min_per_class),
                display_crop_factor=float(display_crop_factor),
                display_scale=int(display_scale),
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "prepare-width-calibration",
    help="Prepare deterministic Phase-5 width-calibration measurement tasks",
    args={
        "handoff_dir": {
            "type": str,
            "required": True,
            "help": "Path to the Phase-5 handoff directory",
        },
        "roads_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the repo-local copied roads GeoPackage",
        },
        "roads_layer": {
            "type": str,
            "required": True,
            "help": "Layer name inside the roads GeoPackage",
        },
        "seed": {
            "type": int,
            "required": True,
            "help": "Random seed for deterministic task generation",
        },
        "crop_size_px": {
            "type": int,
            "required": True,
            "help": "Crop size shown during interactive measurement",
        },
        "out_dir": {
            "type": str,
            "required": True,
            "help": "Output directory for task and manifest artifacts",
        },
        "quota_mode": {
            "type": str,
            "required": False,
            "default": "fixed",
            "help": "Task sampling mode: fixed (legacy) or proportional",
        },
        "sampling_rate": {
            "type": float,
            "required": False,
            "default": 0.05,
            "help": "Per-class sampling rate used when quota_mode=proportional",
        },
        "min_per_class": {
            "type": int,
            "required": False,
            "default": 3,
            "help": "Minimum primary tasks per class for proportional mode",
        },
        "max_per_class": {
            "type": int,
            "required": False,
            "default": 0,
            "help": "Optional cap for primary tasks per class; 0 disables cap",
        },
        "repeat_sampling_rate": {
            "type": float,
            "required": False,
            "default": 0.2,
            "help": "Per-class repeat sampling rate used when quota_mode=proportional",
        },
        "repeat_min_per_class": {
            "type": int,
            "required": False,
            "default": 1,
            "help": "Minimum repeat tasks per class in proportional mode",
        },
    },
)
def prepare_width_calibration_cmd(
    handoff_dir: str,
    roads_gpkg: str,
    roads_layer: str,
    seed: int,
    crop_size_px: int,
    out_dir: str,
    quota_mode: str = "fixed",
    sampling_rate: float = 0.05,
    min_per_class: int = 3,
    max_per_class: int = 0,
    repeat_sampling_rate: float = 0.2,
    repeat_min_per_class: int = 1,
) -> int:
    print(
        json.dumps(
            prepare_width_calibration(
                handoff_dir=handoff_dir,
                roads_gpkg=roads_gpkg,
                roads_layer=roads_layer,
                seed=seed,
                crop_size_px=crop_size_px,
                out_dir=out_dir,
                prompt_for_sync=True,
                quota_mode=quota_mode,
                sampling_rate=float(sampling_rate),
                min_per_class=int(min_per_class),
                max_per_class=int(max_per_class),
                repeat_sampling_rate=float(repeat_sampling_rate),
                repeat_min_per_class=int(repeat_min_per_class),
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "measure-width-calibration",
    help="Open the interactive Phase-5 width-calibration measurement viewer",
    args={
        "handoff_dir": {
            "type": str,
            "required": True,
            "help": "Path to the Phase-5 handoff directory",
        },
        "tasks_csv": {
            "type": str,
            "required": True,
            "help": "Prepared task CSV generated by prepare-width-calibration",
        },
        "out_csv": {
            "type": str,
            "required": True,
            "help": "Measurement CSV written incrementally during the session",
        },
        "display_crop_factor": {
            "type": float,
            "default": DEFAULT_DISPLAY_CROP_FACTOR,
            "help": "Fraction of the prepared crop shown around the anchor point",
        },
        "display_scale": {
            "type": int,
            "default": DEFAULT_DISPLAY_SCALE,
            "help": "Nearest-neighbor display scaling factor for the shown crop",
        },
        "resume": {
            "type": bool,
            "action": "store_true",
            "help": "Resume an existing measurement CSV instead of starting fresh",
        },
    },
)
def measure_width_calibration_cmd(
    handoff_dir: str,
    tasks_csv: str,
    out_csv: str,
    display_crop_factor: float = DEFAULT_DISPLAY_CROP_FACTOR,
    display_scale: int = DEFAULT_DISPLAY_SCALE,
    resume: bool = False,
) -> int:
    print(
        json.dumps(
            measure_width_calibration(
                handoff_dir=handoff_dir,
                tasks_csv=tasks_csv,
                out_csv=out_csv,
                display_crop_factor=display_crop_factor,
                display_scale=display_scale,
                resume=bool(resume),
                prompt_for_sync=True,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "summarize-width-calibration",
    help="Summarize Phase-5 width-calibration measurements",
    args={
        "measurements_csv": {
            "type": str,
            "required": True,
            "help": "Measurement CSV written by measure-width-calibration",
        },
        "out_dir": {
            "type": str,
            "required": True,
            "help": "Output directory for summary artifacts",
        },
    },
)
def summarize_width_calibration_cmd(
    measurements_csv: str,
    out_dir: str,
) -> int:
    print(
        json.dumps(
            summarize_width_calibration(
                measurements_csv=measurements_csv,
                out_dir=out_dir,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "audit-width-calibration-sensitivity",
    help="Run the mask-level sensitivity audit for Phase-5 width calibration",
    args={
        "summary_csv": {
            "type": str,
            "required": True,
            "help": "Summary CSV written by summarize-width-calibration",
        },
        "handoff_dir": {
            "type": str,
            "required": True,
            "help": "Path to the Phase-5 handoff directory",
        },
        "roads_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the repo-local copied roads GeoPackage",
        },
        "out_dir": {
            "type": str,
            "required": True,
            "help": "Output directory for sensitivity artifacts",
        },
    },
)
def audit_width_calibration_sensitivity_cmd(
    summary_csv: str,
    handoff_dir: str,
    roads_gpkg: str,
    out_dir: str,
) -> int:
    print(
        json.dumps(
            audit_width_calibration_sensitivity(
                summary_csv=summary_csv,
                handoff_dir=handoff_dir,
                roads_gpkg=roads_gpkg,
                out_dir=out_dir,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


@cli_command(
    "render-width-calibration-debug-masks",
    help="Render debug/test-only Phase-5 patch masks with one fixed width for all classes",
    args={
        "handoff_dir": {
            "type": str,
            "required": True,
            "help": "Path to the Phase-5 handoff directory",
        },
        "roads_gpkg": {
            "type": str,
            "required": True,
            "help": "Path to the roads GeoPackage used for debug rendering",
        },
        "out_dir": {
            "type": str,
            "required": True,
            "help": "Output directory for debug/test-only mask GeoTIFFs",
        },
        "fixed_width_px": {
            "type": int,
            "required": True,
            "help": "Fixed full road width in pixels used for every class",
        },
        "roads_layer": {
            "type": str,
            "required": False,
            "default": None,
            "help": "Optional layer name inside the roads GeoPackage; defaults to automatic resolution",
        },
    },
)
def render_width_calibration_debug_masks_cmd(
    handoff_dir: str,
    roads_gpkg: str,
    out_dir: str,
    fixed_width_px: int,
    roads_layer: str | None = None,
) -> int:
    print(
        json.dumps(
            render_width_calibration_debug_masks(
                handoff_dir=handoff_dir,
                roads_gpkg=roads_gpkg,
                out_dir=out_dir,
                fixed_width_px=fixed_width_px,
                roads_layer=roads_layer,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0
