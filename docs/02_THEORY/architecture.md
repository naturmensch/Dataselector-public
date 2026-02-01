# Architecture Overview

This document will describe the main components and data flow in the pipeline:

- Input: `data/new_all_tiles.csv` + `data/images/`
- Feature extraction: `src/feature_extractor.py` (DINOv2 / ResNet50)
- Dimensionality reduction: UMAP
- Clustering: `src/clustering.py` (KMeans)
- Selection: `src/multi_criteria_facility_location.py` and `src/spatial_facility_location.py`
- Orchestration: `scripts/run_thesis_pipeline.py`, `scripts/xxl_KDR146_run_thesis_complete_modern.py`

(Full architecture doc to be expanded with diagrams and module relationships.)