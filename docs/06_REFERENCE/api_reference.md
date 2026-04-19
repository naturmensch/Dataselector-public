# API Reference (Active Surface)

This page documents the active Python API surface that is intended for direct
use in scripts, notebooks, and integrations.

## Import Rules

1. Prefer `dataselector.*` import paths.
2. Treat legacy `src/*` imports as non-authoritative compatibility paths.
3. For pipeline execution, prefer the CLI (`python -m dataselector <command>`)
   over direct calls into internal workflow helpers.

## Top-Level Public API

From [dataselector/__init__.py](../../dataselector/__init__.py):

1. `dataselector.load_tiles`
2. `dataselector.load_or_compute_features`
3. `dataselector.MetadataProcessor`
4. `dataselector.FeatureExtractor`
5. `dataselector.ClusteringPipeline`
6. `dataselector.DiversitySelector`
7. `dataselector.MultiCriteriaFacilityLocation`
8. `dataselector.SpatialConstrainedFacilityLocation`
9. `dataselector.compute_metrics`
10. `dataselector.Visualizer`

Export model:
1. The package root re-exports selected symbols from `data`, `features`,
   `analysis`, and `selection`.
2. Prefer importing from `dataselector` or `dataselector.<domain>`.
3. Avoid depending on deep private modules unless you are extending internals.

## Data APIs

### `dataselector.data.load.load_tiles`

```python
load_tiles(
	csv: str | Path,
	image_dir: str | Path = "data/images",
	*,
	prefer_shortname: bool = True,
	fill_missing_image_paths: bool = True,
	missing_placeholder: str = "missing_placeholder.png",
) -> TileSet
```

Purpose:
1. Load metadata CSV.
2. Enrich temporal metadata.
3. Resolve image paths.
4. Return canonical `TileSet` container.

### `dataselector.data.load.load_metadata_df`

```python
load_metadata_df(
	csv: str | Path,
	image_dir: str | Path = "data/images",
	*,
	prefer_shortname: bool = True,
) -> pd.DataFrame
```

Purpose: convenience wrapper if you only need a DataFrame and no `TileSet`.

### `dataselector.data.tiles.TileSet`

Canonical data container passed across the feature and selection pipeline.

Typical fields include:
1. `df`
2. `metadata_csv`
3. `image_dir`
4. `provenance`

## Feature APIs

### `dataselector.features.pipeline.load_or_compute_features`

```python
load_or_compute_features(
	tiles: TileSet,
	*,
	out_dir: str | Path = "outputs",
	batch_size: int = 16,
	cache: bool = True,
) -> np.ndarray
```

Purpose:
1. Compute or load cached feature embeddings aligned with `tiles.df`.
2. Use canonical cache behavior via the active data I/O layer.

### `dataselector.features.feature_extractor.FeatureExtractor`

Primary class for feature extraction configuration and execution.
Use this when you need model-level control beyond the convenience pipeline call.

## Domain Package Exports

### `dataselector.data`

From [dataselector/data/__init__.py](../../dataselector/data/__init__.py):

1. `TileSet`
2. `metadata_processor`
3. `MetadataProcessor`
4. `load_tiles`
5. `TileSampler`
6. `RasterCoverage`

### `dataselector.features`

From [dataselector/features/__init__.py](../../dataselector/features/__init__.py):

1. `FeatureExtractor`
2. `load_or_compute_features`
3. `EmbeddingCache`

### `dataselector.selection`

From [dataselector/selection/__init__.py](../../dataselector/selection/__init__.py):

1. `DiversitySelector`
2. `MultiCriteriaFacilityLocation`
3. `SpatialConstrainedFacilityLocation`
4. `selection_pipeline`
5. `min_distance_pipeline`
6. `apply_spatial_mask`

### `dataselector.analysis`

From [dataselector/analysis/__init__.py](../../dataselector/analysis/__init__.py):

1. `ClusteringPipeline`
2. `compute_metrics`
3. `Visualizer`

## Selection APIs

### `dataselector.selection.DiversitySelector`

Primary selector abstraction for thesis-relevant diverse sampling.

### `dataselector.selection.MultiCriteriaFacilityLocation`

Implements weighted multi-criteria facility-location style selection.

### `dataselector.selection.SpatialConstrainedFacilityLocation`

Facility-location variant with spatial hard-constraint enforcement.

## Analysis APIs

### `dataselector.analysis.ClusteringPipeline`

Pipeline helper for dimensionality reduction and clustering workflows.

### `dataselector.analysis.compute_metrics`

Computes evaluation metrics for selected sets.

### `dataselector.analysis.visualizer.Visualizer`

Visualization helper for analysis and reporting surfaces.

## Workflow APIs (Advanced)

These APIs are callable, but operational execution should usually go through the
CLI commands documented in [CLI_COMMAND_CATALOG.md](CLI_COMMAND_CATALOG.md).

### `dataselector.workflows.thesis_pipeline.run_thesis_pipeline`

Main thesis workflow entrypoint used by the `thesis-pipeline` command.

Signature:

```python
run_thesis_pipeline(config: str | Path = "config/pipeline_config.yaml") -> None
```

### `dataselector.workflows.thesis_orchestrate.run_thesis_orchestrate`

Scientific trigger-all orchestration used by `thesis-orchestrate`.

Signature:

```python
run_thesis_orchestrate(config: str | Path, output_dir: str | Path, force: bool = False) -> None
```

### Width calibration workflow functions

Available under:
1. [dataselector/workflows/width_calibration/](../../dataselector/workflows/width_calibration/)

Prefer the dedicated CLI commands for operational use.

## Canonical Execution

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

## Related References

1. [CLI command catalog](CLI_COMMAND_CATALOG.md)
2. [Administrative tools reference](TOOLS_REFERENCE.md)
3. [Thesis pipeline how-to](../03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
4. [Config policy](../08_GOVERNANCE/CONFIG_POLICY.md)

## Stability Notes

1. Public imports listed on this page are the intended stable surface for
	integrations.
2. Workflow internals in `dataselector.workflows.*` can evolve faster than root
	package exports.
3. For production execution, prefer CLI contracts over direct workflow calls.
