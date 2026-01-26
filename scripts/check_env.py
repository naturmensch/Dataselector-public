#!/usr/bin/env python3
"""Check important native dependency versions for integration tests.

Exits with code 0 when environment is compatible, otherwise non-zero and prints
clear diagnostic information.
"""
import sys
from packaging.version import Version, InvalidVersion

REQUIRED_NUMPY_EXACT = "1.26.4"
REQUIRED_NUMBA_EXACT = "0.63.1"

results = {}

try:
    import numpy as np
    results['numpy'] = np.__version__
except Exception as e:
    print(f"ERROR: Failed to import numpy: {e}")
    sys.exit(2)

# Check NumPy version (require exact pinned version for reproducibility)
if results['numpy'] != REQUIRED_NUMPY_EXACT:
    print(f"INCOMPATIBLE: NumPy {results['numpy']} != {REQUIRED_NUMPY_EXACT} (required exact)")
    print("Hint: run './scripts/exec_in_env.sh --env dataselector --create --ensure-packages \"numpy==1.26.4 numba==0.63.1\" --yes -- <cmd>' to fix your env")
    sys.exit(3)

print(f"numpy: {results['numpy']} (OK)")

# numba
try:
    import numba
    if numba.__version__ != REQUIRED_NUMBA_EXACT:
        print(f"INCOMPATIBLE: Numba {numba.__version__} != {REQUIRED_NUMBA_EXACT} (recommended exact)")
        sys.exit(4)
    print(f"numba: {numba.__version__}")
except Exception as e:
    print(f"numba import error: {e}")
    # numba is optional for some tests, but fail for full E2E
    sys.exit(4)

# numba
try:
    import numba
    print(f"numba: {numba.__version__}")
except Exception as e:
    print(f"numba import error: {e}")
    # numba is optional for some tests, but warn and fail for full E2E
    sys.exit(4)

# umap
try:
    import umap
    print(f"umap: {umap.__version__}")
except Exception as e:
    print(f"umap import error: {e}")
    # If umap fails, many pipeline parts will fail
    sys.exit(5)

# apricot
try:
    import apricot
    print("apricot: present")
except Exception as e:
    print(f"apricot import error: {e}")
    sys.exit(6)

print("Environment checks passed — OK for full integration/E2E runs")
sys.exit(0)
