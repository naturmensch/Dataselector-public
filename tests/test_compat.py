import importlib
import sys
from types import SimpleNamespace

import pytest


def _patch_modules(
    monkeypatch,
    numpy_version="1.26.4",
    numba_version="0.63.1",
    umap_present=True,
    apricot_present=True,
    prefix_contains_env=True,
):
    # Patch numpy
    numpy_mod = SimpleNamespace(__version__=numpy_version)
    monkeypatch.setitem(sys.modules, "numpy", numpy_mod)

    # Patch numba
    numba_mod = SimpleNamespace(__version__=numba_version)
    monkeypatch.setitem(sys.modules, "numba", numba_mod)

    # Patch umap
    if umap_present:
        umap_mod = SimpleNamespace(__version__="0.5.11")
        monkeypatch.setitem(sys.modules, "umap", umap_mod)
    else:
        sys.modules.pop("umap", None)

    # Patch apricot
    if apricot_present:
        apricot_mod = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "apricot", apricot_mod)
    else:
        sys.modules.pop("apricot", None)

    # Patch sys.prefix to a controlled fake value so tests are deterministic
    base_prefix = "/fake/prefix"
    if prefix_contains_env:
        monkeypatch.setattr(sys, "prefix", base_prefix + "/envs/dataselector")
    else:
        monkeypatch.setattr(sys, "prefix", base_prefix + "/envs/other")


@pytest.mark.unit
def test_validate_environment_full_pass(monkeypatch):
    _patch_modules(monkeypatch)
    import dataselector.compat as compat

    res = compat.validate_environment_full(raise_on_error=True)
    # critical fields should be True
    assert res["numpy"] is True
    assert res["numba"] is True
    assert res["env_name"] is True
    # optional fields
    assert res["umap"] is True
    assert res["apricot"] is True


@pytest.mark.unit
def test_validate_environment_full_numpy_mismatch(monkeypatch):
    _patch_modules(monkeypatch, numpy_version="2.4.0")
    import importlib

    import dataselector.compat as compat

    with pytest.raises(RuntimeError) as exc:
        compat.validate_environment_full(raise_on_error=True)
    assert "NumPy version mismatch" in str(exc.value) or "NumPy" in str(exc.value)


@pytest.mark.unit
def test_validate_environment_full_envname_mismatch(monkeypatch):
    _patch_modules(monkeypatch, prefix_contains_env=False)
    import dataselector.compat as compat

    with pytest.raises(RuntimeError) as exc:
        compat.validate_environment_full(raise_on_error=True)
    assert "Not running in required conda env" in str(exc.value)
