from __future__ import annotations

import random
from pathlib import Path

import pytest

from dataselector.runtime.repro_mode import activate_repro_mode


def test_activate_repro_mode_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown execution profile"):
        activate_repro_mode(profile="unknown", seed=1)


def test_thesis_repro_sets_thread_limits_and_seeds(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    state = activate_repro_mode(profile="thesis_repro", seed=123)

    assert state["profile"] == "thesis_repro"
    assert state["seed"] == 123

    # Deterministic RNG behavior for Python random and NumPy.
    first_py = random.random()
    activate_repro_mode(profile="thesis_repro", seed=123)
    second_py = random.random()
    assert first_py == second_py

    import numpy as np

    first_np = np.random.rand(3)
    activate_repro_mode(profile="thesis_repro", seed=123)
    second_np = np.random.rand(3)
    assert (first_np == second_np).all()

    assert state["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert state["thread_env"]["NUMBA_NUM_THREADS"] == "1"


def test_default_profile_keeps_contract_metadata():
    state = activate_repro_mode(profile="default", seed=7)
    assert state["profile"] == "default"
    assert state["seed"] == 7


def test_thesis_repro_marks_parallelism_degraded_when_dev_shm_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    original_exists = Path.exists

    def _fake_exists(self: Path) -> bool:
        if str(self) == "/dev/shm":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _fake_exists)
    state = activate_repro_mode(profile="thesis_repro", seed=123)
    assert state["parallelism_degraded"] is True
    assert state["repro_degraded"] is True
    assert any("dev_shm_missing" in w for w in state["repro_warnings"])


def test_thesis_repro_interop_reinit_error_is_not_false_positive(
    monkeypatch: pytest.MonkeyPatch,
):
    torch = pytest.importorskip("torch")
    from dataselector.runtime import repro_mode as repro_mod

    def _raise_known_interop_error(_value: int) -> None:
        raise RuntimeError(
            "cannot set number of interop threads after parallel work has started "
            "or set_num_interop_threads called"
        )

    class _DummyTempFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        repro_mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: _DummyTempFile(),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", _raise_known_interop_error)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)

    state = activate_repro_mode(profile="thesis_repro", seed=123)
    assert state["repro_degraded"] is False
    assert state["parallelism_degraded"] is False
    assert "set_num_interop_threads_already_initialized" in state["repro_warnings"]
