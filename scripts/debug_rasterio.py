<<<<<<< HEAD
#!/usr/bin/env python3
"""Debug RasterIO.

This script helps debug RasterIO-related issues in the Dataselector project,
checking file access, format support, and common problems.
"""

import sys
from pathlib import Path
import traceback
from scripts.common import DATA_DIR, data_path

ROOT = Path(__file__).resolve().parents[1]


def check_rasterio_import():
    """Check RasterIO import."""
    print("Checking RasterIO import...")
    try:
        import rasterio
        print(f"✅ RasterIO version: {rasterio.__version__}")
        return True
    except ImportError as e:
        print(f"❌ RasterIO import failed: {e}")
        return False


def check_gdal_installation():
    """Check GDAL installation."""
    print("\nChecking GDAL installation...")
    try:
        from osgeo import gdal
        print(f"✅ GDAL version: {gdal.VersionInfo()}")
        return True
    except ImportError as e:
        print(f"❌ GDAL import failed: {e}")
        return False


def test_rasterio_basic():
    """Test basic RasterIO functionality."""
    print("\nTesting basic RasterIO functionality...")
    try:
        import rasterio
        from rasterio.crs import CRS

        # Test CRS creation
        crs = CRS.from_epsg(4326)
        print(f"✅ CRS creation: {crs}")

        # Determine available drivers (adapt to rasterio API changes)
        drivers = None
        try:
            drivers_attr = rasterio.drivers
            # rasterio.drivers may be a module exposing helper functions
            try:
                # Preferred: use raster_driver_extensions() provided by the drivers module
                drivers = drivers_attr.raster_driver_extensions()
            except Exception:
                # Fallbacks
                if callable(drivers_attr):
                    drivers = drivers_attr()
                elif isinstance(drivers_attr, dict):
                    drivers = drivers_attr
                elif hasattr(rasterio, "supported_drivers"):
                    drivers = rasterio.supported_drivers()
        except Exception:
            drivers = None

        if drivers is None:
            print("⚠️ Could not determine available drivers")
        else:
            try:
                drivers_count = len(drivers)
            except TypeError:
                try:
                    drivers_count = len(list(drivers))
                except Exception:
                    drivers_count = 'unknown'
            print(f"✅ Available drivers: {drivers_count} drivers")

        return True
    except Exception as e:
        print(f"❌ Basic RasterIO test failed: {e}")
        traceback.print_exc()
        return False


def check_sample_files():
    """Check for sample raster files."""
    print("\nChecking for sample raster files...")

    # Look for common raster formats
    extensions = ['.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg']

    data_dir = DATA_DIR
    if data_dir.exists():
        raster_files = []
        for ext in extensions:
            raster_files.extend(list(data_dir.rglob(f'*{ext}')))

        if raster_files:
            print(f"✅ Found {len(raster_files)} raster files:")
            for f in raster_files[:5]:  # Show first 5
                print(f"   {f}")
            if len(raster_files) > 5:
                print(f"   ... and {len(raster_files) - 5} more")
            return True
        else:
            print("⚠️  No raster files found in data/ directory")
            return False
    else:
        print("❌ data/ directory not found")
        return False


def test_raster_opening():
    """Test opening a sample raster file."""
    print("\nTesting raster file opening...")

    import rasterio
    from pathlib import Path

    # Try to find a test file
    data_dir = DATA_DIR
    test_files = list(data_dir.rglob('*.tif')) + list(data_dir.rglob('*.tiff'))

    if test_files:
        test_file = test_files[0]
        print(f"Trying to open: {test_file}")

        try:
            with rasterio.open(test_file) as src:
                print("✅ Successfully opened raster file")
                print(f"   Driver: {src.driver}")
                print(f"   Size: {src.width} x {src.height}")
                print(f"   Bands: {src.count}")
                print(f"   CRS: {src.crs}")
                print(f"   Transform: {src.transform}")

                # Try reading a small sample
                data = src.read(1, window=((0, min(10, src.height)), (0, min(10, src.width))))
                print(f"   Sample data shape: {data.shape}")
                print(f"   Data type: {data.dtype}")
                print(f"   Value range: {data.min()} - {data.max()}")

            return True
        except Exception as e:
            print(f"❌ Failed to open raster file: {e}")
            traceback.print_exc()
            return False
    else:
        print("⚠️  No test raster files found")
        return False


def check_common_issues():
    """Check for common RasterIO issues."""
    print("\nChecking for common issues...")

    # Check environment variables
    import os
    env_vars = ['GDAL_DATA', 'PROJ_LIB', 'GDAL_DRIVER_PATH']

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set")

    # Check if conda environment is used
    if 'CONDA_PREFIX' in os.environ:
        print(f"✅ Conda environment: {os.environ['CONDA_PREFIX']}")
    else:
        print("⚠️  Not in a conda environment")

    # Check Python path for conflicts and rasterio installation path
    try:
        import rasterio as _r
        print(f"✅ RasterIO module location: {_r.__file__}")
    except Exception:
        python_path = sys.path
        rasterio_paths = [p for p in python_path if 'rasterio' in (p or '').lower()]
        if rasterio_paths:
            print("✅ RasterIO-like paths in Python path:")
            for p in rasterio_paths:
                print(f"   {p}")
        else:
            print("⚠️  RasterIO not found in Python path")

    return True


def run_debug():
    """Run complete RasterIO debug."""
    print("🐛 RasterIO Debug Tool")
    print("=" * 30)

    checks = [
        check_rasterio_import,
        check_gdal_installation,
        test_rasterio_basic,
        check_sample_files,
        test_raster_opening,
        check_common_issues
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check {check.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 30)
    print("📊 Debug Summary:")

    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")

    if passed == total:
        print("✅ All checks passed!")
    else:
        print("⚠️  Some issues found. Check output above.")

    print("\n💡 Troubleshooting tips:")
    print("   1. Ensure GDAL and PROJ are installed")
    print("   2. Use conda: conda install gdal rasterio")
    print("   3. Check environment variables")
    print("   4. Try reinstalling rasterio")


if __name__ == '__main__':
    run_debug()
=======
from pathlib import Path

from rasterio import open as ropen

p = Path("data/images/KDR_327.png")
print("file exists:", p.exists())
try:
    with ropen(p) as ds:
        print("bounds:", ds.bounds)
        print("crs:", ds.crs)
        print("w,h:", ds.width, ds.height)
except Exception as e:
    print("error opening with rasterio:", e)
>>>>>>> chore/ci-lint-attrs-gdf
