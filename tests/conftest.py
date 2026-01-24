import os
import sys

# Ensure repository root is on sys.path so `src` can be imported in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import types
import sys
import pytest

REPO_ROOT_PATH = Path(ROOT)


@pytest.fixture(scope="session")
def repo_root():
    """Returns the Path object to the repository root."""
    return REPO_ROOT_PATH


# --- Shared fixtures for consolidation PoC ---
@pytest.fixture
def make_dummy_metadata():
    """Return a factory that creates deterministic metadata DataFrames."""
    import pandas as _pd
    import numpy as _np

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
    original_keys = {k for k in sys.modules if k.startswith("src.")}

    yield sys.modules["src"]

    # Cleanup: remove any new src.* modules created during the test
    current_keys = {k for k in sys.modules if k.startswith("src.")}
    for k in current_keys - original_keys:
        del sys.modules[k]

def pytest_sessionstart(session):
    """Fail early if tests are not running inside the project's conda env.

    This prevents accidental test runs in `base` or other environments where
    native dependencies (Numba/UMAP/Apricot) are incompatible. It gives a
    clear actionable message to run tests via `./scripts/exec_in_env.sh`.
    """
    import os, sys

    # Allow bypass for automation or debugging (set this env var to 1 to override)
    if os.environ.get("DATASELECTOR_IGNORE_ENV_CHECK") == "1":
        return

    # Check common conda env indicators
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("MAMBA_DEFAULT_ENV")
    if env_name == "dataselector" or ("dataselector" in sys.prefix):
        return

    # Not in the correct env — abort with helpful instructions
    msg = (
        "Tests must be run inside the 'dataselector' environment.\n"
        "To run the curated integration suite locally, use:\n"
        "  ./scripts/exec_in_env.sh --env dataselector -- pytest -q -m integration\n"
        "Or to run a single test file: \n"
        "  ./scripts/exec_in_env.sh --env dataselector -- pytest -q <path/to/test_file.py>\n"
        "If you understand the implications and want to bypass this check for debugging, set:\n"
        "  export DATASELECTOR_IGNORE_ENV_CHECK=1\n"
    )
    import pytest

    pytest.exit(msg, returncode=2)
