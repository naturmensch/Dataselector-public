from __future__ import annotations

import random

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
