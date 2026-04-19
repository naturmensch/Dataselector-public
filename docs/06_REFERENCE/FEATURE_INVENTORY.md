# Feature Inventory

This document summarizes the retained active and secondary-active CLI surface.

## Canonical thesis commands

| Command | Owner | Role |
|---|---|---|
| `dataselector thesis-orchestrate` | `dataselector.workflows.thesis_orchestrate` | Canonical end-to-end thesis orchestration |
| `dataselector thesis-pipeline` | `dataselector.workflows.thesis_pipeline` | Canonical direct pipeline run |
| `dataselector generate-monitor` | `dataselector.workflows.generate_reports` | Thesis-run summary from `outputs/runs/<run_id>` |

## Retained secondary-active commands

| Command | Owner | Role |
|---|---|---|
| `dataselector thesis-sampler-suite` | `dataselector.workflows.thesis_sampler_suite` | Supplementary sampler comparison |
| `dataselector autoscale` | `dataselector.workflows.autoscale` | Parameter-search helper |
| `dataselector compare-samplers` | `dataselector.workflows.compare_samplers` | Comparative analysis |
| `dataselector benchmark-sampling` | `dataselector.workflows.benchmark_sampling` | Exploration benchmark |

## Removed legacy workflow surface

The former long-run XXL workflow, its dedicated monitor command, and the old
resume / recovery CLI story are no longer active package capabilities.

Historical documentation for that surface lives under
`docs/07_ARCHIVE/legacy_xxl_ops/`.
