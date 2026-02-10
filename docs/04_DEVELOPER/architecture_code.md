# Code Architecture (Deep Dive)

This page tracks the current module-level architecture in `dataselector/`.

## Core Modules

1. `dataselector/data/io.py` - metadata loading and feature cache plumbing
2. `dataselector/features/feature_extractor.py` - embedding extraction
3. `dataselector/selection/clustering.py` - clustering/UMAP helpers
4. `dataselector/selection/multi_criteria_facility_location.py` - selection logic
5. `dataselector/pipeline/experiment_manager.py` - run manifests and provenance
6. `dataselector/workflows/thesis_pipeline.py` - canonical thesis orchestration

## Legacy Note

References to `src/*` are historical only and should not be used for active
development.
