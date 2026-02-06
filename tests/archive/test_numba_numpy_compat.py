import sys
import types

import pytest
from src import compat


def test_no_numba_installed(monkeypatch):
    # Ensure numba is not present: simulate ImportError on import to avoid
    # manipulating sys.modules directly (which can leave the package in a
    # partially-initialized state for subsequent tests).
    monkeypatch.delenv("NUMBA_TEST_DUMMY", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("No module named 'numba'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert compat.check_numba_numpy_compatibility() is True


def test_numba_with_compatible_numpy(monkeypatch, request):
    # Simulate numba present and numpy version 2.3.1
    fake_numba = types.SimpleNamespace(__version__="0.63.1")
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    class FakeNumpy:
        __version__ = "2.3.1"

    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)

    def _reload_real_numba():
        try:
            import importlib

            importlib.reload(importlib.import_module("numba"))
        except Exception:
            pass

    request.addfinalizer(_reload_real_numba)

    assert compat.check_numba_numpy_compatibility() is True


def test_numba_with_incompatible_numpy(monkeypatch, request):
    # Simulate numba present and numpy version 2.4.0
    fake_numba = types.SimpleNamespace(__version__="0.63.1")
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    class FakeNumpy:
        __version__ = "2.4.0"

    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)

    def _reload_real_numba():
        try:
            import importlib

            importlib.reload(importlib.import_module("numba"))
        except Exception:
            pass

    request.addfinalizer(_reload_real_numba)

    with pytest.raises(RuntimeError):
        compat.check_numba_numpy_compatibility()
