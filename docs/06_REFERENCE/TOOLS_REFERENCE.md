# Administrative Tools Reference

The `dataselector tools` subcommands provide utility functions for workspace management, validation, and auditing.

## Available Tools

### `dataselector tools check-geo`

**Purpose:** Validate GIS dependencies and environment configuration.

**Description:**  
Verifies that all required geospatial dependencies (geopandas, pyproj, shapely, fiona, rtree, gdal) are installed and correctly configured.

**Usage:**
```bash
dataselector tools check-geo
```

**Output:**  
- ✅ Installed versions of each GIS library
- ⚠️ Warnings if optional libraries (rasterio) are missing
- ❌ Errors if critical dependencies fail

**Example Output:**
```
✓ geopandas: 0.11.1
✓ shapely: 2.0.1
✓ pyproj: 3.4.1
✓ fiona: 1.8.21
⚠ rasterio: not available (optional, for geotiff reading)
✓ All critical GIS dependencies available
```

---

### `dataselector tools align-audit`

**Purpose:** Check CSV vs. Raster alignment and validate spatial consistency.

**Description:**  
Analyzes alignment between metadata CSV coordinates and raster images. Verifies:
- CRS (Coordinate Reference System) consistency
- Bounding box overlaps
- Pixel-to-world coordinate transforms

**Usage:**
```bash
dataselector tools align-audit \
  --csv data/new_all_tiles.csv \
  --image-dir data/images/ \
  --output outputs/alignment_report.json
```

**Output:**  
- JSON report with alignment metrics
- CSV with per-tile alignment status
- Visualization plot (if matplotlib available)

---

### `dataselector tools protect-paths`

**Purpose:** Verify that protected paths are not accidentally committed.

**Description:**  
Audits git staging area to ensure that critical files/directories (large images, raw data, final outputs) cannot be accidentally committed. Useful for preventing repository bloat.

**Protected Paths (Default):**
```
data/images/
data/archive/
data/raw/
models/
outputs/final_selection/
outputs/kdr100_selection/
```

**Usage:**
```bash
# Check currently staged files
dataselector tools protect-paths

# Add custom protected paths
PROTECTED_PATHS="data/large_datasets,outputs/temp" dataselector tools protect-paths
```

**Output:**
- ✅ "All staged files are safe" (no protected paths detected)
- ❌ "Offending files detected: [list]" (if protected paths are staged)

**Exit Code:**
- 0: All safe
- 1: Protected paths detected

---

### `dataselector tools audit-files`

**Purpose:** Comprehensive workspace security and configuration audit.

**Description:**  
Performs systematic checks on workspace integrity:
- Protected file verification
- Environment variable configuration
- Dependency pins consistency
- Archive integrity

**Usage:**
```bash
dataselector tools audit-files

# With report output
dataselector tools audit-files --report outputs/audit_report.txt
```

**Output:**
- Summary of all checks (passed/failed)
- Detailed findings for each category
- Recommendations for issues found

---

### `dataselector tools archive-list`

**Purpose:** List and describe available archived artifacts.

**Description:**  
Discovers and catalogs all archived outputs in `data/archive/`. Helps users understand what historical experiment outputs are available for restoration.

**Usage:**
```bash
dataselector tools archive-list

# With detailed metadata
dataselector tools archive-list --detailed

# Filter by date
dataselector tools archive-list --since 2026-01-01
```

**Output:**
```
Archive: outputs_archive_20260112_153935.tar.gz
  Size: 2.3 GB
  Created: 2026-01-12 15:39:35
  Contains: phase_0/, phase_1/, outputs/
  Description: Complete thesis pipeline outputs before cleanup

Archive: outputs_archive_20260201_102015.tar.gz
  Size: 1.8 GB
  Created: 2026-02-01 10:20:15
  Contains: tuning_weights/, bootstrap/
  Description: Optimization and bootstrap results
```

---

### `dataselector tools cleanup`

**Purpose:** Clean up temporary and regenerable artifacts from workspace.

**Description:**  
Safely removes non-critical files to reclaim disk space:
- Temporary directories (`outputs/tmp_*`)
- Conda/venv directories
- Cache files (feature extractions, clustering caches)
- Old experiment runs

**Usage:**
```bash
# Dry-run (preview what will be deleted)
dataselector tools cleanup --dry-run

# Remove only temporary directories
dataselector tools cleanup --remove-tmp

# Remove everything regenerable (features, caches, venvs)
dataselector tools cleanup --aggressive

# Preserve specific artifacts
dataselector tools cleanup --preserve "outputs/tuning_weights,outputs/final_selection"
```

**Output:**
```
Dry-run mode:
  Would remove: outputs/tmp_feature_extraction/ (234 MB)
  Would remove: mambafhyc7nt0lcj/ (512 MB)
  Would remove: outputs/cache/ (45 MB)
Total space to reclaim: 791 MB
```

---

### `python -m dataselector docs-link-check`

**Purpose:** Validate relative documentation links on the active docs surface.

**Description:**  
Scans Markdown documentation for broken relative file links.
- Default gate: the active/authoritative `docs/` surface
- Historical/generated `docs/reports/` is excluded by default
- Use `--include-historical` for an optional full scan including historical docs
- When you pass a concrete docs subdirectory in code, that subtree is scanned fully

**Usage:**
```bash
# Check the active docs surface
micromamba run -n dataselector -- python -m dataselector docs-link-check

# Optional deep diagnosis including historical/generated docs
micromamba run -n dataselector -- \
  python -m dataselector docs-link-check --include-historical
```

**Output:**
```
✓ No broken links found

✗ Found 2 broken links:
  - docs/INDEX.md: Legacy note -> 07_ARCHIVE/missing_note.md
  - docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md: Policy -> ../missing.md
```

**Exit Code:**
- 0: All links valid
- 1: Broken links found

---

### `python -m dataselector docs-link-autofix`

**Purpose:** Attempt to auto-fix broken relative documentation links.

**Description:**  
Attempts basename-based fixes for broken relative links in Markdown docs.
- Default scope matches `docs-link-check`
- Use `--include-historical` to also consider `docs/reports/`
- Dry-run by default; apply changes only with `--yes`
- Optional backups are written unless `--no-backup` is set

**Usage:**
```bash
# Preview candidate fixes on the active docs surface
micromamba run -n dataselector -- python -m dataselector docs-link-autofix

# Apply fixes and keep backups
micromamba run -n dataselector -- python -m dataselector docs-link-autofix --yes

# Include historical/generated docs in the scan
micromamba run -n dataselector -- \
  python -m dataselector docs-link-autofix --include-historical
```

**Output:**
```
Would fix: docs/INDEX.md
  ../old_reference.md -> ../06_REFERENCE/new_reference.md

[DRY RUN] Would fix 1 links automatically
Would require manual fix: 2 links
```

---

### `dataselector tools deps-check`

**Purpose:** Verify dependency compatibility and pin versions.

**Description:**  
Validates that all dependencies (Python packages, system libraries, optional extensions) are compatible with the project. Useful for CI/CD and environment setup verification.

**Usage:**
```bash
# Check all dependencies
dataselector tools deps-check

# With detailed versions
dataselector tools deps-check --verbose

# Check only critical dependencies
dataselector tools deps-check --critical
```

**Output:**
```
Python Environment: 3.11.7
PyTorch: 2.0.1 (CUDA 11.8) ✓
NumPy: 1.24.3 ✓
Pandas: 2.0.1 ✓
GIS Stack:
  geopandas: 0.11.1 ✓
  pyproj: 3.4.1 ✓
  shapely: 2.0.1 ✓

⚠ Warning: GDAL version 3.6.0 is older than recommended 3.8.0
  (functionality OK, but some features may be slower)

✓ All dependencies compatible
```

---

## Common Workflows

### Verify workspace before committing
```bash
dataselector tools protect-paths && \
python -m dataselector docs-link-check && \
dataselector tools deps-check
```

### Clean workspace after large experiments
```bash
dataselector tools cleanup --dry-run
dataselector tools cleanup --aggressive --preserve "outputs/final_selection"
```

### Full audit before sharing workspace
```bash
dataselector tools audit-files
dataselector tools align-audit --csv data/new_all_tiles.csv --image-dir data/images
dataselector tools check-geo
```

### Post-migration validation
```bash
python -m dataselector docs-link-autofix
python -m dataselector docs-link-check
dataselector tools deps-check --verbose
```

---

## Error Handling

Most tools provide helpful error messages and recovery suggestions:

**Example: Missing GIS dependencies**
```
Error: geopandas not found. Please install the geo stack:
  pip install -r requirements-geo.txt
Or:
  mamba env create -f environment.yml
```

**Example: Protected path violation**
```
Error: The following files are in protected paths and cannot be committed:
  data/images/kdr100_50000.tif
  data/archive/outputs_archive_20260201.tar.gz

Fix: Remove these files from git staging
  git reset HEAD data/images/* data/archive/*
```

---

## Performance Notes

- `align-audit`: 2-5 minutes for full KDR100 dataset (depends on image count)
- `docs-link-check`: <1 second for small documentation sets, ~10 seconds for full docs
- `cleanup`: 1-5 minutes depending on cleanup scope
- `audit-files`: ~30 seconds for comprehensive check

For large workspaces, consider running tools with `--partial` or date filters for faster results.
