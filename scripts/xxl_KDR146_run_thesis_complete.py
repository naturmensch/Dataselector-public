#!/usr/bin/env python3
"""
Legacy shim: Delegates to xxl_KDR146_run_thesis_complete_modern.py
This file is maintained for backward compatibility with existing tests and scripts.
All new code should import from the modern version.
"""

import importlib.util
import sys
from pathlib import Path

# Load the modern version as a shim
ROOT = Path(__file__).resolve().parents[1]
modern_path = ROOT / "scripts" / "xxl_KDR146_run_thesis_complete_modern.py"

spec = importlib.util.spec_from_file_location(
    "xxl_KDR146_run_thesis_complete_modern",
    modern_path,
)
modern_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modern_module)

# Re-export all public functions and classes from modern version
_extract_xxl_final_statistics = modern_module._extract_xxl_final_statistics
_find_xxl_runs = modern_module._find_xxl_runs
_validate_convergence_from_validation_data = modern_module._validate_convergence_from_validation_data
phase_1_xxl_hamburg = modern_module.phase_1_xxl_hamburg
phase_2_reproducibility = modern_module.phase_2_reproducibility
run_cmd_with_retry = modern_module.run_cmd_with_retry

# If this script is run directly, delegate to main
if __name__ == "__main__":
    sys.exit(modern_module.main())
