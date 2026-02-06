import os
import sys
from pathlib import Path

import numpy as np

# --- PFAD-SETUP: MUSS GANZ OBEN STEHEN ---
# Ermittelt den absoluten Pfad zum Projekt-Root (ein Level über 'tests')
# und fügt ihn an Position 0 in sys.path ein, falls noch nicht vorhanden.
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# Versuche, das Paket früh zu importieren, damit `from src.xxx import ...`
# beim Sammeln (collect) funktioniert.
try:
    import src  # type: ignore
except Exception:
    # Falls Import trotzdem scheitert, lassen wir pytest die eigentliche Fehlermeldung anzeigen
    pass
# -----------------------------------------

import types
from pathlib import Path

import pytest
from _pytest.config import Config

REPO_ROOT_PATH = Path(root_dir)


# --- CLI decorator registration ---
# Import all CLI modules to trigger @cli_command decorators
# This ensures _CLI_COMMANDS is populated before tests run
def _register_all_cli_commands():
    """Import all CLI modules to populate _CLI_COMMANDS registry."""
    try:
        # Import all workflow modules
        # Import data module
        import dataselector.data.build_tiles
        import dataselector.tools.archive
        import dataselector.tools.audit

        # Import all tools modules
        import dataselector.tools.check
        import dataselector.tools.clean
        import dataselector.tools.docs_link
        import dataselector.workflows.adaptive_pipeline
        import dataselector.workflows.autoscale
        import dataselector.workflows.benchmark_sampling
        import dataselector.workflows.bootstrap
        import dataselector.workflows.compare_samplers
        import dataselector.workflows.final_selection
        import dataselector.workflows.generate_reports
        import dataselector.workflows.optuna_optimize
        import dataselector.workflows.sampler_suite
        import dataselector.workflows.thesis_pipeline
        import dataselector.workflows.thesis_sampler_suite
        import dataselector.workflows.xxl
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
        return _pd.DataFrame(
            {
                "N": rng.uniform(48, 55, n),
                "left": rng.uniform(6, 15, n),
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
        lines = ["longName,shortName,N,left,image_path,image_filename,year"]
        for i in range(n):
            lines.append(f"A{i},a{i},{50.0 + i*0.1},{10 + i*0.1},,,{1900+i}")
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
    """Warn if pytest is not run via exec_in_env.sh and validate the test environment.

    - If not invoked via `exec_in_env.sh`, emit a clear warning.
    - If invoked via `exec_in_env.sh`, run `scripts/check_env.py` and record its outcome.
      E2E tests will be automatically skipped later when the environment check fails.
    """
    import subprocess

    from _pytest.config import Config

    # Warn if not using the canonical wrapper
    if not os.environ.get("EXEC_IN_ENV"):
        import warnings

        warnings.warn(
            "pytest wird nicht über exec_in_env.sh ausgeführt! "
            "Verwende 'make test' oder './scripts/exec_in_env.sh -- pytest ...' "
            "um sicherzustellen, dass die korrekte Umgebung verwendet wird.",
            UserWarning,
            stacklevel=2,
        )
        # Mark environment as unchecked/invalid for E2E purposes
        config._env_check_ok = False
        config._env_check_msg = "EXEC_IN_ENV not set; run tests via './scripts/exec_in_env.sh --env dataselector -- pytest ...'"
        return

    # If we are inside the wrapper, run the environment diagnostic script to ensure compatibility
    try:
        res = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "check_env.py"),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            config._env_check_ok = True
            config._env_check_msg = "Environment check passed"
        else:
            config._env_check_ok = False
            # Provide actionable message (stdout/stderr may contain details)
            out = (res.stdout or "").strip()
            err = (res.stderr or "").strip()
            config._env_check_msg = (
                "Environment check failed: \n" + out + "\n" + err + "\n"
                "Run: './scripts/exec_in_env.sh --env dataselector --create --ensure-packages "
                "numpy==1.26.4 numba==0.63.1"
                " --yes' to fix."
            )
    except Exception as e:
        config._env_check_ok = False
        config._env_check_msg = f"Failed to run environment check: {e}"


def pytest_collection_modifyitems(config: Config, items):
    """Skip E2E-marked tests when environment check failed.

    This keeps E2E tests honest (they are only executed in a compatible environment)
    without resorting to local monkeypatches to hide import errors.
    """
    if getattr(config, "_env_check_ok", False):
        return

    skip_reason = getattr(
        config, "_env_check_msg", "Environment not suitable for E2E tests"
    )
    skip_marker = pytest.mark.skip(reason=skip_reason)

    for item in list(items):
        if "e2e" in {m.name for m in item.iter_markers()}:
            item.add_marker(skip_marker)
