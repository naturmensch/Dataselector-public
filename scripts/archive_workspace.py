#!/usr/bin/env python3
"""Workspace archiving utility: Clean up old experiments, redundant scripts, and temporary files.

Usage:
    python scripts/archive_workspace.py --dry-run  # Show what would be archived
    python scripts/archive_workspace.py --category scripts  # Archive only deprecated scripts
    python scripts/archive_workspace.py --all --yes  # Archive everything non-interactively
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "archive_local"

# Whitelist: NEVER archive these
WHITELIST_PATTERNS = {
    ".git",
    ".github",
    "data/images",
    "data/new_all_tiles.csv",
    "config/pipeline_config.yaml",
    "requirements.txt",
    "requirements-cpu.txt",
    "README.md",
    "pyproject.toml",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "archive_local",
    "scripts/deprecated",  # Already archived
    "outputs/final_selection",  # Never archive final selections
}

# Files to keep in outputs (recent/important)
OUTPUTS_KEEP = {
    "features.npy",
    "metadata.csv",
    "coarse_sweep",
    "fine_sweep",
    "feasibility_analysis",
    "final_selection",
}


def is_whitelisted(path: Path) -> bool:
    """Check if path matches whitelist patterns."""
    rel_path = path.relative_to(ROOT)
    path_str = str(rel_path)
    
    for pattern in WHITELIST_PATTERNS:
        if path_str.startswith(pattern) or pattern in path_str:
            return True
    return False


def get_file_age_days(path: Path) -> int:
    """Get file age in days."""
    if not path.exists():
        return 0
    mtime = path.stat().st_mtime
    age_seconds = datetime.now().timestamp() - mtime
    return int(age_seconds / 86400)


class ArchiveCategory:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.files: List[Path] = []
        self.total_size = 0

    def add_file(self, path: Path):
        if path.exists() and path.is_file():
            self.files.append(path)
            self.total_size += path.stat().st_size

    def add_directory(self, path: Path):
        if path.exists() and path.is_dir():
            for file in path.rglob("*"):
                if file.is_file() and not is_whitelisted(file):
                    self.add_file(file)

    def summary(self) -> str:
        size_mb = self.total_size / (1024 * 1024)
        return f"{self.name}: {len(self.files)} files, {size_mb:.1f} MB"


def identify_archive_candidates() -> Dict[str, ArchiveCategory]:
    """Identify all files/directories to archive."""
    categories = {
        "scripts": ArchiveCategory("deprecated_scripts", "Legacy/redundant scripts"),
        "outputs": ArchiveCategory("old_outputs", "Old experiment outputs (>30 days)"),
        "docs": ArchiveCategory("doc_fragments", "Outdated documentation fragments"),
        "configs": ArchiveCategory("old_configs", "Obsolete config variants"),
        "notebooks": ArchiveCategory("old_notebooks", "Outdated/exploratory notebooks"),
    }

    # 1. SCRIPTS: Legacy/redundant scripts
    legacy_scripts = [
        "scripts/tune_weights_and_run.py",
        "scripts/run_diverse_experiments.py",
        "scripts/multi_criteria_temporal_test.py",
        "scripts/compare_methods.py",
        "scripts/benchmark_speed.py",
        "scripts/quick_benchmark.py",
        "scripts/run_pipeline.py",
        "scripts/profile_selection.py",
        "scripts/plot_seeded_vs_unseeded.py",
        "scripts/aggregate_validation_seed_vs_unseed.py",
        "scripts/check_protected.py",
        "scripts/manage_archives.py",
        "scripts/clean_workspace.py",
        "scripts/generate_reports.py",
        "scripts/install_git_hooks.py",
    ]
    for script in legacy_scripts:
        path = ROOT / script
        if path.exists() and not is_whitelisted(path):
            categories["scripts"].add_file(path)

    # 2. OUTPUTS: Old experiment runs (keep recent, archive old)
    outputs_dir = ROOT / "outputs"
    if outputs_dir.exists():
        for item in outputs_dir.iterdir():
            if is_whitelisted(item):
                continue
            
            # Keep explicitly protected outputs
            if item.name in OUTPUTS_KEEP:
                continue
            
            # Archive old experiment folders
            if item.is_dir():
                age = get_file_age_days(item)
                
                # Keep experiments < 7 days, archive older ones
                if "experiments" in item.name or "validation" in item.name:
                    if age > 7:
                        categories["outputs"].add_directory(item)
                
                # Archive specific old folders
                elif item.name in [
                    "tuning_weights",
                    "optuna_comparison",
                    "optuna_comparison_v2",
                    "optuna_comparison_v3",
                    "test_run",
                    "test_vis",
                    "seed_benchmark",
                    "validation_fine",
                    "validation_fine_seeded",
                    "validation_seeded",
                    "cache_backup_20260112",
                ]:
                    categories["outputs"].add_directory(item)
            
            # Archive specific old files
            elif item.is_file():
                if item.name in [
                    "debug_findings.md",
                    "debug_test.csv",
                    "features_test.npy",
                    "optimization_analysis.csv",
                    "optimization_report.md",
                    "optimization_results.csv",
                    "optimization_results_subset_50.csv",
                    "optimized_parameters.csv",
                    "profile_legacy.prof",
                    "profile_legacy.txt",
                    "profile_constraint_integrated.prof",
                    "profile_constraint_integrated.txt",
                    "profile_multi_criteria.prof",
                    "profile_multi_criteria.txt",
                    "profile_summary.csv",
                    "quick_benchmark_summary.json",
                    "method_comparison.csv",
                    "multi_criteria_temporal_test.csv",
                    "multi_criteria_full_selection.csv",
                ]:
                    categories["outputs"].add_file(item)

    # 3. DOCS: Outdated documentation fragments
    old_docs = [
        "docs/fine_sweep_report.md",
        "docs/parameter_determination.md",
        "docs/seed_benchmark.md",
        "outputs/scientific_solution.md",
        "outputs/pipeline_analysis.md",
        "outputs/report_20260112.md",
        "outputs/final_optimization_report.md",
        "CLEANUP_PLAN.md",
        "todo260115.md",
    ]
    for doc in old_docs:
        path = ROOT / doc
        if path.exists() and not is_whitelisted(path):
            categories["docs"].add_file(path)

    # 4. CONFIGS: Obsolete config variants
    old_configs = [
        "outputs/optuna_autoscale_best_20260112.json",
        "outputs/optuna_autoscale_report_20260112.md",
        "outputs/optuna_autoscale_summary_20260112.csv",
    ]
    for cfg in old_configs:
        path = ROOT / cfg
        if path.exists() and not is_whitelisted(path):
            categories["configs"].add_file(path)

    # 5. NOTEBOOKS: Outdated/exploratory notebooks
    notebooks_dir = ROOT / "notebooks"
    if notebooks_dir.exists():
        for nb in notebooks_dir.glob("*.ipynb"):
            age = get_file_age_days(nb)
            # Keep recent notebooks (< 14 days), archive old exploratory ones
            if age > 14 and not is_whitelisted(nb):
                categories["notebooks"].add_file(nb)

    return categories


def create_manifest(category: ArchiveCategory, archive_path: Path):
    """Create manifest file for archived category."""
    manifest = {
        "archived_at": datetime.now().isoformat(),
        "category": category.name,
        "description": category.description,
        "total_files": len(category.files),
        "total_size_mb": round(category.total_size / (1024 * 1024), 2),
        "files": [
            {
                "path": str(f.relative_to(ROOT)),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in category.files
        ],
    }
    
    manifest_path = archive_path / f"{category.name}_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    
    return manifest_path


def archive_category(category: ArchiveCategory, dry_run: bool = True) -> int:
    """Archive files in a category."""
    if not category.files:
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{category.name}_{timestamp}"

    if dry_run:
        print(f"\n[DRY RUN] Would archive to: {archive_path}")
        for i, file in enumerate(category.files[:10], 1):
            print(f"  {i}. {file.relative_to(ROOT)}")
        if len(category.files) > 10:
            print(f"  ... and {len(category.files) - 10} more files")
        return len(category.files)

    # Create archive directory
    archive_path.mkdir(parents=True, exist_ok=True)

    # Copy files preserving structure
    archived_count = 0
    for file in category.files:
        rel_path = file.relative_to(ROOT)
        dest = archive_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(file, dest)
            archived_count += 1
        except Exception as e:
            print(f"Warning: Failed to archive {rel_path}: {e}")

    # Create manifest
    manifest_path = create_manifest(category, archive_path)
    print(f"✓ Archived {archived_count} files to {archive_path}")
    print(f"✓ Manifest: {manifest_path}")

    return archived_count


def main():
    parser = argparse.ArgumentParser(description="Archive old/redundant workspace files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be archived without doing it")
    parser.add_argument("--category", choices=["scripts", "outputs", "docs", "configs", "notebooks", "all"], 
                        default="all", help="Category to archive")
    parser.add_argument("--yes", action="store_true", help="Non-interactive mode (assume yes)")
    parser.add_argument("--delete-after-archive", action="store_true", 
                        help="Delete original files after archiving (DANGEROUS)")
    args = parser.parse_args()

    print("=" * 80)
    print("WORKSPACE ARCHIVING UTILITY")
    print("=" * 80)
    print(f"Root: {ROOT}")
    print(f"Archive destination: {ARCHIVE_DIR}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Identify candidates
    print("Analyzing workspace...")
    categories = identify_archive_candidates()

    # Filter by category
    if args.category != "all":
        categories = {k: v for k, v in categories.items() if k == args.category}

    # Show summary
    print("\nArchive Summary:")
    print("-" * 80)
    total_files = 0
    total_size = 0
    for cat in categories.values():
        if cat.files:
            print(f"  {cat.summary()}")
            total_files += len(cat.files)
            total_size += cat.total_size
    
    total_size_mb = total_size / (1024 * 1024)
    print("-" * 80)
    print(f"TOTAL: {total_files} files, {total_size_mb:.1f} MB")
    print()

    if not total_files:
        print("Nothing to archive. Workspace is clean! ✨")
        return 0

    # Confirm
    if not args.dry_run and not args.yes:
        response = input(f"Proceed with archiving {total_files} files? [y/N] ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            return 1

    # Archive each category
    archived_total = 0
    for cat in categories.values():
        if cat.files:
            archived_total += archive_category(cat, dry_run=args.dry_run)

    print()
    print("=" * 80)
    if args.dry_run:
        print(f"DRY RUN COMPLETE: Would archive {archived_total} files")
        print("Run without --dry-run to perform actual archiving.")
    else:
        print(f"✓ ARCHIVING COMPLETE: {archived_total} files archived")
        if args.delete_after_archive:
            print("\nDeleting original files...")
            deleted = 0
            for cat in categories.values():
                for file in cat.files:
                    try:
                        file.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"Warning: Failed to delete {file}: {e}")
            print(f"✓ Deleted {deleted} original files")
        else:
            print("Original files preserved. Use --delete-after-archive to remove them.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
