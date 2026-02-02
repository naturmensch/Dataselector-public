#!/usr/bin/env python3
"""
Codemod: Migrate all 'from src.x import' to 'from dataselector.<pkg>.x import'

This script handles the Phase 4 migration of imports from src/ to dataselector/ subpackages.
"""

import re
import sys
from pathlib import Path

# Mapping of src modules to dataselector subpackages
IMPORT_MAPPING = {
    # selection modules
    "from dataselector.selection.clustering import": "from dataselector.selection.clustering import",
    "from dataselector.selection.diversity_selector import": "from dataselector.selection.diversity_selector import",
    "from dataselector.selection.multi_criteria_facility_location import": "from dataselector.selection.multi_criteria_facility_location import",
    "from dataselector.selection.spatial_facility_location import": "from dataselector.selection.spatial_facility_location import",
    "from dataselector.selection.lazy_facility_location import": "from dataselector.selection.lazy_facility_location import",
    "from dataselector.selection.pareto import": "from dataselector.selection.pareto import",
    # data modules
    "from dataselector.data.io import": "from dataselector.data.io import",
    "from dataselector.data.metadata_processor import": "from dataselector.data.metadata_processor import",
    # features modules
    "from dataselector.features.feature_extractor import": "from dataselector.features.feature_extractor import",
    # pipeline modules
    "from dataselector.pipeline.experiment_manager import": "from dataselector.pipeline.experiment_manager import",
    "from dataselector.pipeline.experiments import": "from dataselector.pipeline.experiments import",
    "from dataselector.pipeline.cache import": "from dataselector.pipeline.cache import",
    "from dataselector.pipeline.pipeline_utils import": "from dataselector.pipeline.pipeline_utils import",
    "from dataselector.pipeline.incremental_results import": "from dataselector.pipeline.incremental_results import",
    # analysis modules
    "from dataselector.analysis.metrics import": "from dataselector.analysis.metrics import",
    "from dataselector.analysis.visualizer import": "from dataselector.analysis.visualizer import",
    "from dataselector.analysis.wandb_logger import": "from dataselector.analysis.wandb_logger import",
    # workflows modules
    "from dataselector.workflows.sampling_strategies import": "from dataselector.workflows.sampling_strategies import",
}

# Special cases: import src.X or import src.X as Y
IMPORT_AS_MAPPING = {
    "import dataselector.selection.clustering": "import dataselector.selection.clustering",
    "import dataselector.selection.diversity_selector": "import dataselector.selection.diversity_selector",
    "import dataselector.selection.multi_criteria_facility_location": "import dataselector.selection.multi_criteria_facility_location",
    "import dataselector.selection.spatial_facility_location": "import dataselector.selection.spatial_facility_location",
    "import dataselector.selection.lazy_facility_location": "import dataselector.selection.lazy_facility_location",
    "import dataselector.selection.pareto": "import dataselector.selection.pareto",
    "import dataselector.data.io": "import dataselector.data.io",
    "import dataselector.data.metadata_processor": "import dataselector.data.metadata_processor",
    "import dataselector.features.feature_extractor": "import dataselector.features.feature_extractor",
    "import dataselector.pipeline.experiment_manager": "import dataselector.pipeline.experiment_manager",
    "import dataselector.pipeline.experiments": "import dataselector.pipeline.experiments",
    "import dataselector.pipeline.cache": "import dataselector.pipeline.cache",
    "import dataselector.pipeline.pipeline_utils": "import dataselector.pipeline.pipeline_utils",
    "import dataselector.pipeline.incremental_results": "import dataselector.pipeline.incremental_results",
    "import dataselector.analysis.metrics": "import dataselector.analysis.metrics",
    "import dataselector.analysis.visualizer": "import dataselector.analysis.visualizer",
    "import dataselector.analysis.wandb_logger": "import dataselector.analysis.wandb_logger",
    "import dataselector.workflows.sampling_strategies": "import dataselector.workflows.sampling_strategies",
}

def migrate_file(filepath):
    """Migrate imports in a single file."""
    try:
        content = filepath.read_text()
        original = content
        
        # Apply all mappings
        for old, new in IMPORT_MAPPING.items():
            content = content.replace(old, new)
        
        for old, new in IMPORT_AS_MAPPING.items():
            content = content.replace(old, new)
        
        if content != original:
            filepath.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"ERROR: {filepath}: {e}", file=sys.stderr)
        return False

def main():
    repo_root = Path(__file__).parents[1]
    exclude_dirs = {'.git', 'archive', 'archive_local', 'tmp', 'locks', '.egg-info', '__pycache__', 'src'}
    
    changed_files = []
    total_files = 0
    
    for py_file in sorted(repo_root.rglob("*.py")):
        # Skip excluded directories
        if any(exc in py_file.parts for exc in exclude_dirs):
            continue
        
        total_files += 1
        if migrate_file(py_file):
            changed_files.append(py_file.relative_to(repo_root))
    
    print(f"Migrated {len(changed_files)} files out of {total_files}")
    for f in changed_files:
        print(f"  ✓ {f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
