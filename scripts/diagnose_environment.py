#!/usr/bin/env python3
"""Diagnose Environment.

This script diagnoses the environment for geospatial and data processing
capabilities, checking for required libraries and system configurations.
"""

import sys
import platform
import subprocess
from pathlib import Path
import importlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def check_python_version():
    """Check Python version."""
    print("🐍 Python Version Check:")
    print(f"   Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Architecture: {platform.architecture()}")
    print()


def check_package(package_name: str, import_name: str = None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name}: Not found ({e})")
        return False


def check_geospatial_packages():
    """Check geospatial packages."""
    print("🗺️  Geospatial Packages:")

    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('geopandas', 'geopandas'),
        ('shapely', 'shapely'),
        ('rasterio', 'rasterio'),
        ('pyproj', 'pyproj'),
        ('fiona', 'fiona'),
        ('matplotlib', 'matplotlib'),
        ('cartopy', 'cartopy'),
    ]

    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False

    print()
    return all_ok


def check_ml_packages():
    """Check machine learning packages."""
    print("🤖 Machine Learning Packages:")

    packages = [
        ('scikit-learn', 'sklearn'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('optuna', 'optuna'),
        ('umap-learn', 'umap'),
        ('hdbscan', 'hdbscan'),
        ('scipy', 'scipy'),
    ]

    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False

    print()
    return all_ok


def check_system_commands():
    """Check system commands."""
    print("💻 System Commands:")

    commands = ['gdal_translate', 'gdalinfo', 'proj', 'cs2cs']

    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"   ✅ {cmd}: {version}")
            else:
                print(f"   ❌ {cmd}: Failed to get version")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"   ❌ {cmd}: Not found")

    print()


def check_environment_variables():
    """Check relevant environment variables."""
    print("🔧 Environment Variables:")

    import os
    vars_to_check = ['PYTHONPATH', 'PROJ_LIB', 'GDAL_DATA', 'PATH']

    for var in vars_to_check:
        value = os.environ.get(var, 'Not set')
        if len(str(value)) > 50:
            value = str(value)[:47] + '...'
        print(f"   {var}: {value}")

    print()


def check_data_paths():
    """Check data paths."""
    print("📁 Data Paths:")

    paths_to_check = [
        ROOT / 'data',
        ROOT / 'outputs',
        ROOT / 'notebooks',
        ROOT / 'config'
    ]

    for path in paths_to_check:
        if path.exists():
            if path.is_dir():
                try:
                    files = list(path.glob('*'))
                    print(f"   ✅ {path.name}/: {len(files)} items")
                except PermissionError:
                    print(f"   ⚠️  {path.name}/: Permission denied")
            else:
                print(f"   ✅ {path.name}: File exists")
        else:
            print(f"   ❌ {path.name}: Not found")

    print()


def run_full_diagnosis():
    """Run complete environment diagnosis."""
    print("🔍 Environment Diagnosis for Dataselector")
    print("=" * 50)

    check_python_version()
    geo_ok = check_geospatial_packages()
    ml_ok = check_ml_packages()
    check_system_commands()
    check_environment_variables()
    check_data_paths()

    print("📊 Summary:")
    if geo_ok and ml_ok:
        print("   ✅ Environment looks good for Dataselector!")
    else:
        print("   ⚠️  Some packages are missing. Check requirements.txt")
        print("      Run: pip install -r requirements.txt")

    print("\n💡 Recommendations:")
    print("   - Ensure GDAL/PROJ are installed for geospatial operations")
    print("   - Use conda/mamba for package management")
    print("   - Check docs for detailed setup instructions")


if __name__ == '__main__':
    run_full_diagnosis()