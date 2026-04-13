# Phase-5 Width Calibration Method

## Purpose

This workflow calibrates classwise pixel widths for the binary Phase-5 road-mask
generation step. It is designed for the actual 54-patch handoff and supports a
deterministic measurement queue, an interactive two-click measurement tool, and
per-class summary plus mask-level sensitivity outputs.

## Scope

- Use only the actual 54-patch repo-local handoff.
- Exclude Hamburg `KDR_146_p1/p2` at task-generation time.
- Run the workflow from repo-local inputs only after the roads GeoPackage has
  been copied into a repo-local ignored path.
- Keep the downstream training stack binary. This calibration workflow does not
  introduce multiclass training.

## Commands

Build the canonical repo-local Phase-5 roads source from the classified base
layer plus the tracer-derived class-4 and class-5 layers:

```bash
micromamba run -n dataselector python -m dataselector build-width-calibration-roads-source \
  --cut-roads-gpkg <qgis-project-root>/cut_fixed_geometry_roads.gpkg \
  --tracer4-gpkg <qgis-project-root>/4_roads_tracer_patches.gpkg \
  --tracer5-gpkg <qgis-project-root>/5_roads_tracer_patches.gpkg
```

This writes the canonical merged source to
`handoff/local_sources/phase5_roads_merged.gpkg` with layer
`phase5_roads_merged` plus provenance in
`handoff/local_sources/phase5_roads_merged.sources.json`.

`sync-width-calibration-source` remains available only as an auxiliary/manual
copy path when you already have a single editable GeoPackage that should be
mirrored into an ignored repo-local location. It is not the canonical Phase-5
width-calibration input anymore.

Prepare deterministic measurement tasks:

```bash
micromamba run -n dataselector python -m dataselector prepare-width-calibration \
  --handoff-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core \
  --roads-gpkg handoff/local_sources/phase5_roads_merged.gpkg \
  --roads-layer phase5_roads_merged \
  --seed 42 \
  --crop-size-px 64 \
  --out-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px
```

If sync metadata already exists for the repo-local roads copy and the editable
QGIS source has changed, `prepare-width-calibration` will offer a terminal
confirmation prompt to sync first before generating a new queue.

If the target width-calibration run directory already contains a stale run,
`prepare-width-calibration` will also offer to archive the full existing run to
`<out_dir>_archive_<UTCSTAMP>` before rebuilding the active directory.

Open the interactive measurement queue:

```bash
micromamba run -n dataselector python -m dataselector measure-width-calibration \
  --handoff-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core \
  --tasks-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_tasks.csv \
  --out-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_measurements.csv
```

If the editable QGIS source has changed since the current queue was prepared,
`measure-width-calibration` will offer a terminal confirmation prompt to sync
the repo-local copy first. After such a sync it stops intentionally and asks
you to rerun `prepare-width-calibration`, because the active queue is then
stale by definition.

`measure-width-calibration` never archives stale queues automatically.

Resume an existing measurement session:

```bash
micromamba run -n dataselector python -m dataselector measure-width-calibration \
  --handoff-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core \
  --tasks-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_tasks.csv \
  --out-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_measurements.csv \
  --resume
```

Generate the class summary:

```bash
micromamba run -n dataselector python -m dataselector summarize-width-calibration \
  --measurements-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_measurements.csv \
  --out-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px
```

Run the mask-level sensitivity audit:

```bash
micromamba run -n dataselector python -m dataselector audit-width-calibration-sensitivity \
  --summary-csv handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px/width_calibration_summary.csv \
  --handoff-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core \
  --roads-gpkg handoff/local_sources/phase5_roads_merged.gpkg \
  --out-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core_widths_64px
```

`summarize-width-calibration` now writes only per-class summary statistics.
`audit-width-calibration-sensitivity` writes only the mask-level sensitivity
artifacts and overlays.

Render fixed-width debug/test-only patch masks for a minimal downstream smoke:

```bash
micromamba run -n dataselector python -m dataselector render-width-calibration-debug-masks \
  --handoff-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core \
  --roads-gpkg handoff/local_sources/phase5_roads_merged.gpkg \
  --out-dir handoff/thesis_orchestrate_20260313T200624Z_patches_core_debug_masks_10px \
  --fixed-width-px 10
```

This path is explicitly `debug_only` / `test_only`.
It does not replace width calibration, does not change any scientific artifact,
and must not be used for final thesis masks.

## Measurement Definition

- Single-line classes use the full visible stroke width.
- Double-line classes use the total visible road signature including the inner
  gap.
- Dashed classes are measured on one representative dash only, never across a
  gap.
- Width is measured approximately orthogonal to the local line direction.

Rejected spots must be coded as one of:

- `crossing`
- `label_overlap`
- `endpoint`
- `tight_curve`
- `blur_damage`
- `ambiguous_symbol`
- `crop_too_small`
- `click_error`
- `other`

## Sampling Logic

- Candidate anchors are generated automatically from the roads layer and the
  georeferenced quicklooks.
- The human does not choose which object comes next.
- Common classes `0`, `1`, `6` target 12 primary tasks each and cover 4
  distinct tiles before same-tile duplication.
- Mid-frequency classes `2`, `5` target 9 primary tasks each and cover 3
  distinct tiles before same-tile duplication.
- Class `9` uses all eligible anchors up to a target of 9 primary tasks.
- Classes `3`, `4`, and `8` keep all eligible anchors.
- Repeat tasks are pre-scheduled and shown later with prior values hidden.

## Fixed Eligibility Parameters

The workflow records its fixed eligibility constants in
`width_calibration_manifest.json`.

Current defaults:

- `endpoint_exclusion_fraction = 0.20`
- `minimum_border_margin_factor = 0.50`
- `minimum_in_crop_line_support_px = 32`
- `anchor_positions = (0.10, 0.30, 0.50, 0.70, 0.90)`

## Interactive Measurement

The viewer shows one crop at a time, centered on the prepared anchor point.
By default it displays the full prepared crop and enlarges it with
nearest-neighbor scaling to make symbols easier to inspect.
The interactive viewer requires a Qt 6 backend and uses a Qt-based control
strip plus dialog-driven reject flow.

- Left-click twice to record the visible outer width endpoints.
- `r` rejects the current task and prompts for a coded reason.
- Reject runs fully inside the GUI; it does not use terminal input.
- `s` defers the current task to later in the same session.
- `u` undoes the last recorded measurement.
- `q` quits the viewer safely.
- `Esc` clears the current click pair.

Repeat measurements are used only for reliability assessment and do not
contribute to the final class median.

## Artifacts

Stable documentation:

- `docs/03_USER_GUIDES/PHASE5_WIDTH_CALIBRATION_METHOD.md`

Run artifacts:

- `width_calibration_manifest.json`
- `width_calibration_tasks.csv`
- `width_calibration_measurements.csv`
- `width_calibration_summary.csv`
- `width_calibration_summary.json`
- `width_calibration_sensitivity.csv`
- `width_calibration_sensitivity_overlays/`
- `handoff/local_sources/phase5_roads_merged.sources.json`
- optional auxiliary/manual-copy provenance: `handoff/local_sources/cut_fixed_geometry_roads.sync.json`
- `width_calibration_debug_mask_manifest.json` (`debug_only` / `test_only`)

## Recommended Order

1. `build-width-calibration-roads-source`
2. `prepare-width-calibration`
3. `measure-width-calibration`
4. `summarize-width-calibration`
5. `audit-width-calibration-sensitivity`

## Debug/Test-Only Smoke Path

`render-width-calibration-debug-masks` exists only for technical validation of
mask generation and a minimal downstream training smoke.

- It renders one fixed width for all classes.
- It does not use the scientific width-calibration outputs.
- It does not modify `width_calibration_manifest.json`,
  `width_calibration_summary.csv/json`, or `final_width_px`.
- It is not valid for final thesis masks.

## Summary Outputs

The per-class summary reports:

- `n_valid_primary`
- `median_px`
- `IQR_px`
- `MAD_px`
- `repeat_median_abs_diff_px`
- `low_evidence_flag`
- `high_variance_flag`
- `low_reliability_flag`
- `final_width_px`

The final rasterization width per class is the rounded median of accepted
primary measurements only.

## Sensitivity Audit

The sensitivity audit renders masks for:

- baseline widths
- `median-1px`
- `median+1px`

For each audit patch it reports:

- foreground pixel count
- connected-component count
- saved overlay panels

The audit subset is deterministic and must include every rare class present in
the handoff plus two common-class patches where available.
