import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from _pytest.config import Config

# --- PFAD-SETUP: MUSS GANZ OBEN STEHEN ---
# Ermittelt den absoluten Pfad zum Projekt-Root (ein Level über 'tests')
# und fügt ihn an Position 0 in sys.path ein, falls noch nicht vorhanden.
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# Keine stille Import-Probe hier: Fehler sollen im Collect normal sichtbar werden.
# -----------------------------------------

REPO_ROOT_PATH = Path(root_dir)


# --- CLI decorator registration ---
# Import all CLI modules to trigger @cli_command decorators
# This ensures _CLI_COMMANDS is populated before tests run
def _register_all_cli_commands():
    """Import all CLI modules to populate _CLI_COMMANDS registry."""
    try:
        import importlib

        modules = [
            "dataselector.data.build_tiles",
            "dataselector.tools.archive",
            "dataselector.tools.audit",
            "dataselector.tools.check",
            "dataselector.tools.clean",
            "dataselector.tools.docs_link",
            "dataselector.workflows.adaptive_pipeline",
            "dataselector.workflows.autoscale",
            "dataselector.workflows.benchmark_sampling",
            "dataselector.workflows.bootstrap",
            "dataselector.workflows.compare_samplers",
            "dataselector.workflows.final_selection",
            "dataselector.workflows.generate_reports",
            "dataselector.workflows.optuna_optimize",
            "dataselector.workflows.sampler_suite",
            "dataselector.workflows.thesis_pipeline",
            "dataselector.workflows.thesis_sampler_suite",
            "dataselector.workflows.xxl",
        ]
        for module_name in modules:
            importlib.import_module(module_name)
    except ImportError:
        pass  # Let pytest show the actual error


# Register commands once at module import time
_register_all_cli_commands()
# -----------------------------------------


@pytest.fixture(scope="session")
def repo_root():
    """Returns the Path object to the repository root."""
    return REPO_ROOT_PATH


# --- Shared fixtures for consolidation PoC ---
@pytest.fixture
def make_dummy_metadata():
    """Return a factory that creates deterministic metadata DataFrames."""
    import numpy as _np
    import pandas as _pd

    def _make(n, seed: int = 0):
        rng = _np.random.RandomState(seed)
        center_x = rng.uniform(450000, 650000, n)
        center_y = rng.uniform(5800000, 6100000, n)
        half_w = rng.uniform(40, 80, n)
        half_h = rng.uniform(40, 80, n)
        return _pd.DataFrame(
            {
                "ul_x": center_x - half_w,
                "ul_y": center_y + half_h,
                "lr_x": center_x + half_w,
                "lr_y": center_y - half_h,
                "year": rng.randint(1880, 1945, n),
            }
        )

    return _make


@pytest.fixture
def make_features():
    import numpy as _np

    def _make(n, dim=64, seed: int = 0):
        rng = _np.random.RandomState(seed)
        return rng.randn(n, dim)

    return _make


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    data = ws / "data"
    outputs = ws / "outputs"
    data.mkdir(parents=True)
    outputs.mkdir(parents=True)
    return ws


@pytest.fixture
def create_minimal_new_all_tiles_csv():
    def _fn(path, n=5):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "longName,shortName,ul_x,ul_y,lr_x,lr_y,image_path,image_filename,year"
        ]
        for i in range(n):
            cx = 500000.0 + i * 100.0
            cy = 5900000.0 + i * 100.0
            lines.append(f"A{i},a{i},{cx-50},{cy+50},{cx+50},{cy-50},,,{1900+i}")
        path.write_text("\n".join(lines) + "\n")
        return path

    return _fn


@pytest.fixture
def init_tmp_git_repo(tmp_path):
    import subprocess

    def _init():
        # Initialize a git repo with minimal commit
        subprocess.check_call(["git", "init", str(tmp_path)])
        return tmp_path

    return _init


@pytest.fixture
def tmp_dirs(tmp_path):
    """Creates standard directory structure (data, outputs) in tmp_path."""
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    (data / "images").mkdir(parents=True)
    outputs.mkdir(parents=True)
    return {"root": tmp_path, "data": data, "outputs": outputs}


@pytest.fixture
def inject_src_stub(monkeypatch):
    """
    Ensures 'src' package exists in sys.modules and cleans up any
    submodules attached to it during the test.
    """
    # Ensure src package exists
    if "src" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))

    # Track original state of src children to restore/clean after test
    original_keys = {k for k in sys.modules if k.startswith("dataselector")}

    yield sys.modules["src"]

    # Cleanup: remove any new src.* modules created during the test
    current_keys = {k for k in sys.modules if k.startswith("dataselector")}
    for k in current_keys - original_keys:
        del sys.modules[k]


@pytest.fixture
def fake_features():
    """Return a function that generates fake feature arrays mimicking DINOv2 output."""

    def _fake(n_samples, feature_dim=768):
        # Use random features to avoid identical visual distances
        rng = np.random.RandomState(42)
        return rng.randn(n_samples, feature_dim).astype(np.float32)

    return _fake


@pytest.fixture
def stub_feature_extraction(monkeypatch, fake_features):
    """Stub load_or_extract_features to return fake features instead of running extraction."""
    import sys

    import pandas as pd

    def _fake_loader(tmp_path, csv_meta, cache=True, batch_size=16, **kwargs):
        meta = pd.read_csv(csv_meta)
        n = len(meta)
        return fake_features(n)

    # Patch the function in the module
    monkeypatch.setattr(
        sys.modules["dataselector.data.io"], "load_or_extract_features", _fake_loader
    )
    return _fake_loader


def pytest_configure(config):
    """Validate runtime availability and record compatibility status.

    The canonical test surface is package-first (`python -m dataselector ...`).
    E2E tests are skipped only when the CLI runtime itself is unavailable.
    """
    import subprocess

    # Runtime check via canonical package entrypoint.
    try:
        res = subprocess.run(
            [sys.executable, "-m", "dataselector", "--help"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            config._env_check_ok = True
            config._env_check_msg = "Runtime check passed"
        else:
            config._env_check_ok = False
            out = (res.stdout or "").strip()
            err = (res.stderr or "").strip()
            config._env_check_msg = (
                "Runtime check failed: \n" + out + "\n" + err + "\n"
                "Run tests in the dataselector environment (e.g. "
                "'conda run -n dataselector -- python -m pytest ...')."
            )
    except Exception as e:
        config._env_check_ok = False
        config._env_check_msg = f"Failed to run environment check: {e}"


def pytest_collection_modifyitems(config: Config, items):
    """Apply global test collection policies.

    - Skip E2E tests when runtime check fails.
    - Skip E2E tests unless RUN_FULL_INTEGRATION=1 is set.
    - Skip real-image tests unless DATASELECTOR_IMAGE_DIR points to a valid directory.
    """
    env_ok = getattr(config, "_env_check_ok", False)
    env_skip_reason = getattr(
        config, "_env_check_msg", "Environment not suitable for E2E tests"
    )
    env_skip_marker = pytest.mark.skip(reason=env_skip_reason)

    run_full_integration = os.environ.get(
        "RUN_FULL_INTEGRATION", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    integration_skip_marker = pytest.mark.skip(
        reason=(
            "E2E tests require RUN_FULL_INTEGRATION=1 "
            "(set env var to enable full integration execution)"
        )
    )

    # Real image policy:
    # - Tests marked real_images (or legacy real_tiles) require DATASELECTOR_IMAGE_DIR.
    # - This keeps CI image-independent and moves private image usage to explicit local runs.
    image_dir_env = os.environ.get("DATASELECTOR_IMAGE_DIR", "").strip()
    image_dir_ok = bool(image_dir_env and Path(image_dir_env).exists())
    real_images_skip = pytest.mark.skip(
        reason=(
            "real_images test requires DATASELECTOR_IMAGE_DIR to point to a valid local "
            "directory with private image data"
        )
    )

    for item in list(items):
        marker_names = {m.name for m in item.iter_markers()}
        if "real_tiles" in marker_names and "real_images" not in marker_names:
            item.add_marker(pytest.mark.real_images)
            marker_names.add("real_images")

        if "real_images" in marker_names and not image_dir_ok:
            item.add_marker(real_images_skip)

        if "e2e" in {m.name for m in item.iter_markers()}:
            if not env_ok:
                item.add_marker(env_skip_marker)
            elif not run_full_integration:
                item.add_marker(integration_skip_marker)
