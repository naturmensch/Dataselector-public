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

### `dataselector tools docs-check`

**Purpose:** Validate all documentation links and references.

**Description:**  
Scans documentation for broken links, missing references, and outdated information.
- Internal links (file references)
- External links (URLs, GitHub references)
- Code examples (existence of referenced functions/files)

**Usage:**
```bash
# Check all documentation
dataselector tools docs-check

# Check specific file
dataselector tools docs-check docs/REPRODUCIBILITY.md

# Generate report
dataselector tools docs-check --output docs/link_check_report.txt
```

**Output:**
```
✓ docs/REPRODUCIBILITY.md: 45 links checked, all valid
✓ docs/ARCHITECTURE.md: 32 links checked, all valid
❌ README.md:
   Line 45: Broken link to "docs/old_feature_reference.md" (file moved to docs/06_REFERENCE/)
   Line 67: Dead external link "https://github.com/old-repo/..."
```

**Exit Code:**
- 0: All links valid
- 1: Broken links found

---

### `dataselector tools docs-fix`

**Purpose:** Automatically fix common documentation issues.

**Description:**  
Attempts to auto-correct broken links and update references. Handles:
- File moves/renames
- URL changes
- Obsolete script references → CLI commands

**Usage:**
```bash
# Preview changes (dry-run)
dataselector tools docs-fix --dry-run

# Apply fixes
dataselector tools docs-fix --apply

# Fix specific type only
dataselector tools docs-fix --apply --fix-type "script-to-cli"
```

**Output:**
```
Fixed: README.md
  scripts/run_thesis_pipeline.py → dataselector thesis-pipeline
  
Fixed: docs/REPRODUCIBILITY.md
  docs/old_feature_reference.md → docs/06_REFERENCE/FEATURES.md

Summary: 2 files fixed, 5 references updated
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
dataselector tools docs-check && \
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
dataselector tools docs-fix --dry-run
dataselector tools docs-check
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
- `docs-check`: <1 second for small documentation sets, ~10 seconds for full docs
- `cleanup`: 1-5 minutes depending on cleanup scope
- `audit-files`: ~30 seconds for comprehensive check

For large workspaces, consider running tools with `--partial` or date filters for faster results.
