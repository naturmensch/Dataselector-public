# City Contract Baseline And Resolution (2026-02-09)

## Initial Gap (before fix)

1. Canonical metadata rows: `676`.
2. City coverage was incomplete (`667/676` non-empty).
3. Missing keys were:
   - `KDR_046`
   - `KDR_079a`
   - `KDR_155b`
   - `KDR_175`
   - `KDR_406`
   - `KDR_408`
   - `KDR_446`
   - `KDR_601`
   - `KDR_664`

## Root Cause

1. Sidecar `*.aux.xml` files provide CRS/geotransform only, not city names.
2. Enrichment key drift (`KDR_079A` vs `KDR_079a`, variant suffixes like `KDR_155b`).
3. LongName format drift (`o.J`, `ca1893`, dotted-year suffixes).
4. True source gaps for a small remainder (`KDR_046`, `KDR_175`) required curated overrides.

## Implemented Resolution Chain

1. Normalized key matching (`short_norm`, case-insensitive, filename suffix removal).
2. Variant-base fallback (`KDR_155b -> KDR_155`) when exact row is missing.
3. Tolerant longName parser for city extraction.
4. Deterministic backup fill from best `new_all_tiles.backup_*.csv`.
5. Curated `data/city_overrides.csv` for true residuals.
6. Full source trace in `city_source` per row.

## Current Result (after fix)

1. Canonical metadata rows: `676`.
2. `city_non_empty`: `676`.
3. `city_source_non_empty`: `676`.
4. `city == Hamburg`: `1`.
5. `city == Kiel`: `1`.
6. Missing city rows: `0`.
7. Source distribution:
   - `longname_parse`: `673`
   - `manual_override`: `3`

## Validation

1. `tests/test_build_new_all_tiles.py`
2. `tests/test_preselection.py`
3. `tests/unit/test_city_contract.py`

