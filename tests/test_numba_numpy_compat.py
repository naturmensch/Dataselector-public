import sys
import types

import pytest

from src import compat


def test_no_numba_installed(monkeypatch):
    # Ensure numba is not present
    monkeypatch.delenv("NUMBA_TEST_DUMMY", raising=False)
    if "numba" in sys.modules:
        del sys.modules["numba"]

    assert compat.check_numba_numpy_compatibility() is True


def test_numba_with_compatible_numpy(monkeypatch):
    # Simulate numba present and numpy version 2.3.1
    fake_numba = types.SimpleNamespace(__version__="0.63.1")
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    class FakeNumpy:
        __version__ = "2.3.1"

    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)

    assert compat.check_numba_numpy_compatibility() is True


def test_numba_with_incompatible_numpy(monkeypatch):
    # Simulate numba present and numpy version 2.4.0
    fake_numba = types.SimpleNamespace(__version__="0.63.1")
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    class FakeNumpy:
        __version__ = "2.4.0"

    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)

    with pytest.raises(RuntimeError):
        compat.check_numba_numpy_compatibility()
