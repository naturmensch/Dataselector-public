"""Shared fixtures for E2E tests.

This module provides pytest fixtures for E2E testing, including:
- Temporary workspace management
- Sample data generation
- GIS dependency checking
- Auto-cleanup on test teardown
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator

import psutil
import pytest


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary workspace with standard directory structure.
    
    Creates:
    - data/
    - outputs/
    - outputs/runs/
    - config/
    
    Yields the workspace root path.
    Automatically cleaned up after test.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create standard structure
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    (workspace / "outputs" / "runs").mkdir(parents=True, exist_ok=True)
    (workspace / "config").mkdir(parents=True, exist_ok=True)
    
    yield workspace
    
    # Cleanup is automatic with tmp_path fixture


@pytest.fixture
def sample_csv(tmp_workspace: Path) -> Path:
    """Generate a small sample CSV with minimal metadata for testing.
    
    Creates a CSV with:
    - 50 minimal tile records
    - Columns: id, name, year, longitude, latitude, quadrant
    - Valid geographic coordinates (Germany region)
    
    Returns path to the generated CSV.
    """
    csv_file = tmp_workspace / "data" / "sample_tiles.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate minimal CSV with valid coordinates
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "year", "longitude", "latitude", "quadrant"]
        )
        writer.writeheader()
        
        # 50 sample tiles with geographic coordinates
        for i in range(50):
            year = 1900 + (i % 50)  # Years 1900-1949
            lon = 8.0 + (i % 10) * 0.5  # Longitudes 8-12 (Germany)
            lat = 47.0 + (i // 10) * 0.5  # Latitudes 47-51 (Germany)
            
            writer.writerow({
                "id": f"tile_{i:04d}",
                "name": f"Tile_{i}",
                "year": year,
                "longitude": lon,
                "latitude": lat,
                "quadrant": f"Q{(i % 4) + 1}"
            })
    
    return csv_file


@pytest.fixture
def sample_metadata_dict() -> dict:
    """Generate a dictionary with sample metadata for unit tests.
    
    Returns a dict with 100 tile records (more than sample_csv).
    Useful for in-memory testing without file I/O.
    """
    tiles = []
    for i in range(100):
        tiles.append({
            "id": f"tile_{i:05d}",
            "name": f"Tile_{i}",
            "year": 1900 + (i % 100),
            "longitude": 8.0 + (i % 20) * 0.25,
            "latitude": 47.0 + (i // 20) * 0.25,
            "quadrant": f"Q{(i % 4) + 1}",
            "area_km2": 100.0 + i,  # Varying sizes
            "coverage_pct": 50 + (i % 50),  # 50-99% coverage
        })
    
    return {"tiles": tiles, "count": len(tiles)}


@pytest.fixture
def real_tiles_csv() -> Path:
    """Path to real test tiles CSV with actual image metadata.
    
    Contains 5 actual KDR tiles (KDR_001-005) with real GeoTransform coordinates.
    The corresponding PNG and aux.xml files are in tests/fixtures/real_tiles/.
    
    Returns:
        Path to CSV file with real tile metadata
    
    Note:
        This fixture only returns the path. The CSV and image files must already
        exist in the repository under tests/fixtures/real_tiles/
    """
    csv_path = Path(__file__).parent.parent / "fixtures" / "real_tiles_metadata.csv"
    
    if not csv_path.exists():
        pytest.skip(f"Real tiles CSV not found at {csv_path}")
    
    return csv_path


@pytest.fixture(scope="session")
def skip_if_no_gis() -> None:
    """Require GIS dependencies for test.
    
    Use this fixture at the function level to skip tests if geopandas
    is not available:
    
    Example:
        @pytest.mark.usefixtures("skip_if_no_gis")
        def test_geo_operation():
            import geopandas as gpd
            ...
    """
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    pytest.importorskip("pyproj")


@pytest.fixture
def cleanup_on_teardown(tmp_workspace: Path) -> Generator[Path, None, None]:
    """Ensure workspace cleanup even if test fails.
    
    This fixture wraps the standard tmp_workspace and adds explicit
    cleanup logic to handle edge cases.
    
    Yields workspace path, cleans up on teardown.
    """
    yield tmp_workspace
    
    # Force cleanup (in case any files are locked)
    if tmp_workspace.exists():
        try:
            shutil.rmtree(tmp_workspace, ignore_errors=True)
        except Exception as e:
            pytest.warns(UserWarning, f"Could not cleanup {tmp_workspace}: {e}")


@pytest.fixture
def capsys_quiet(capsys):
    """Suppress stdout/stderr during test execution.
    
    Useful for E2E tests that produce verbose output but we only
    want to capture in case of failure.
    
    Returns capsys object for accessing output if needed.
    """
    return capsys


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Convenience fixture for environment variable patching.
    
    Example:
        def test_with_env_var(monkeypatch_env):
            monkeypatch_env.setenv("DATASELECTOR_MODE", "test")
    """
    return monkeypatch


# Markers for test categorization

def pytest_configure(config):
    """Register custom markers for E2E test categorization."""
    config.addinivalue_line(
        "markers",
        "smoke: fast sanity check (runs in <5 min)"
    )
    config.addinivalue_line(
        "markers",
        "integration: integration test combining multiple components (5-15 min)"
    )
    config.addinivalue_line(
        "markers",
        "workflow: complete end-to-end workflow test (10-30+ min)"
    )
    config.addinivalue_line(
        "markers",
        "error: error handling / graceful degradation test"
    )
    config.addinivalue_line(
        "markers",
        "slow: slow test, use --runslow to include"
    )
    config.addinivalue_line(
        "markers",
        "requires_gis: requires GIS dependencies"
    )
    config.addinivalue_line(
        "markers",
        "requires_images: requires sample image files"
    )


@pytest.fixture(autouse=True)
def configure_venv_path(monkeypatch):
    """Automatically set up PATH and PYTHONPATH to use venv.
    
    This fixture runs for every test and ensures subprocess commands
    find the correct Python environment and dataselector CLI tools.
    """
    import os
    import sys
    
    # Get the venv bin directory
    venv_bin = Path(sys.executable).parent
    
    # Prepend venv bin to PATH so we find dataselector, autoscale, etc.
    current_path = os.environ.get("PATH", "")
    new_path = f"{venv_bin}:{current_path}"
    monkeypatch.setenv("PATH", new_path)
    
    # Also ensure PYTHONHOME includes venv
    monkeypatch.setenv("PYTHONHOME", str(venv_bin.parent))


@pytest.fixture
def run_dataselector_cli():
    """Helper fixture to run dataselector CLI commands in subprocess.
    
    This wrapper ensures commands use the correct Python environment.
    
    Usage:
        result = run_dataselector_cli(["autoscale", "--csv", "data.csv"])
        assert result.returncode == 0
    """
    def _run_cli(cmd_args, **kwargs):
        """Run a dataselector CLI command using subprocess.
        
        Args:
            cmd_args: List of command arguments (e.g., ["autoscale", "--csv", "data.csv"])
            **kwargs: Additional arguments to pass to subprocess.run
        
        Returns:
            CompletedProcess object from subprocess.run
        """
        # Ensure we have proper defaults for subprocess.run
        if "capture_output" not in kwargs:
            kwargs["capture_output"] = True
        if "cwd" not in kwargs and "workspace" in kwargs:
            kwargs["cwd"] = str(kwargs.pop("workspace"))
        
        import os
        env = kwargs.get("env", os.environ.copy())
        repo_root = Path(__file__).resolve().parents[2]
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            env["PYTHONPATH"] = f"{repo_root}:{current_pythonpath}"
        else:
            env["PYTHONPATH"] = str(repo_root)
        kwargs["env"] = env
        
        # Prepend 'python -m dataselector' to the command arguments
        full_cmd = [sys.executable, "-m", "dataselector"] + cmd_args
        
        return subprocess.run(full_cmd, **kwargs)
    
    return _run_cli


def pytest_collection_modifyitems(config, items):
    """Add markers and conditions based on test names and fixtures.
    
    Automatically skips tests that require unavailable resources.
    """
    for item in items:
        # Mark tests by directory
        if "smoke" in item.nodeid:
            item.add_marker(pytest.mark.smoke)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "workflow" in item.nodeid or "xxl" in item.nodeid:
            item.add_marker(pytest.mark.workflow)
        elif "error" in item.nodeid:
            item.add_marker(pytest.mark.error)
        
        # Skip GIS tests if dependencies missing
        if "skip_if_no_gis" in item.fixturenames:
            try:
                __import__("geopandas")
            except ImportError:
                item.add_marker(
                    pytest.mark.skip(reason="geopandas not available")
                )


# Session-scoped fixtures for expensive setup

@pytest.fixture(scope="session")
def dataselector_cli_available() -> bool:
    """Check if dataselector CLI is properly installed and available.
    
    Returns True if dataselector can be imported and CLI works,
    False otherwise.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["dataselector", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Get repository root path.
    
    Returns the absolute path to the repository root (parent of tests/).
    """
    return Path(__file__).parent.parent.parent


# Performance/resource management

@pytest.fixture
def resource_monitor(request):
    """Monitor test resource usage (memory, time).
    
    Logs resource usage at end of test.
    """
    import time
    import psutil
    
    process = psutil.Process()
    start_time = time.time()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    mem_delta = end_mem - start_mem
    
    # Log to stdout (captured by pytest)
    print(f"\n📊 Resource usage for {request.node.name}:")
    print(f"   ⏱️  Duration: {duration:.2f} seconds")
    print(f"   💾 Memory delta: {mem_delta:+.1f} MB")


# Timing/performance markers

@pytest.fixture
def slow_test_threshold():
    """Define threshold for "slow" tests in seconds.
    
    Tests exceeding this threshold are logged as slow.
    Default: 30 seconds for E2E tests.
    """
    return 30.0


@pytest.fixture(autouse=True)
def test_timing(request, slow_test_threshold, capsys):
    """Automatically time all tests and warn if slow.
    
    This fixture runs automatically for all tests.
    """
    import time
    
    start = time.time()
    
    yield
    
    elapsed = time.time() - start
    
    if elapsed > slow_test_threshold:
        capsys.write(
            f"⚠️  SLOW TEST: {request.node.name} took {elapsed:.1f}s "
            f"(threshold: {slow_test_threshold}s)\n"
        )
