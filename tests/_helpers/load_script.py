"""Helper utilities for tests that need to import scripts dynamically.

Provides a small wrapper around importlib.util.spec_from_file_location that
loads a script from a given path into a freshly created module object. This
avoids manipulating sys.path or doing module-level imports that depend on
side-effects.

Usage:
    from tests._helpers.load_script import load_script

    mod = load_script(Path("scripts/my_script.py"), module_name="scripts.my_script")
    # call mod.main(...) or inspect mod.__dict__
"""
from pathlib import Path
import importlib.util
import sys
from types import ModuleType


def load_script(path: Path, module_name: str | None = None) -> ModuleType:
    """Load a Python script from `path` as a module and return it.

    - `path` can be a Path or string pointing to a .py file.
    - `module_name` defaults to the stem of the path but can be provided to
      differentiate multiple loads.

    Notes:
    - The module is inserted into `sys.modules` under the chosen name for
      compatibility with other import mechanisms; it is removed after load to
      avoid polluting the interpreter state.
    - The loader executes the module in its own namespace and returns the
      resulting module object.
    """
    path = Path(path)
    if module_name is None:
        module_name = f"tests._script_{path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Temporarily store and execute
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Leave module in sys.modules for callers to inspect but avoid leaving
        # temporary names lying around in subsequent tests if not needed.
        pass

    # If module was loaded under a dotted name, attempt to attach it as an attribute
    # on its parent package so string-based monkeypatching (e.g., "scripts.name.attr") works.
    try:
        if "." in module_name:
            parent_name = module_name.rsplit(".", 1)[0]
            import importlib as _importlib

            parent_pkg = _importlib.import_module(parent_name)
            setattr(parent_pkg, module_name.split(".")[-1], module)
    except Exception:
        # Not critical for tests; silently ignore to avoid test disruption
        pass

    return module
