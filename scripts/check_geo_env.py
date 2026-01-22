"""Quick Geo dependency checker.

Usage: python scripts/check_geo_env.py
Exits with code 0 if all required geo packages are importable; non-zero otherwise.
Prints versions and brief install hints on failure.
"""
from importlib import import_module
import sys

REQS = ["geopandas", "pyproj", "shapely", "fiona", "rtree"]

# Check pipeline config to see whether geo features are enabled. If disabled, we can skip the strict check.
try:
    import yaml
    cfg = yaml.safe_load(open("config/pipeline_config.yaml"))
    geo_enabled = bool(cfg.get("features", {}).get("geo", True))
except Exception:
    geo_enabled = True

if not geo_enabled:
    print("Geo feature disabled in config/pipeline_config.yaml — skipping geo dependency check.")
    sys.exit(0)

failures = []
for pkg in REQS:
    try:
        m = import_module(pkg)
        ver = getattr(m, "__version__", None)
        print(f"{pkg}: OK (version={ver})")
    except Exception as e:
        print(f"{pkg}: MISSING ({e})")
        failures.append(pkg)

if failures:
    print("\nMissing geo dependencies: ", ", ".join(failures))
    print("Install with conda: conda install -n dataselector -c conda-forge geopandas pyproj shapely fiona rtree rasterio")
    sys.exit(2)
else:
    print("All geo dependencies available.")
    sys.exit(0)
