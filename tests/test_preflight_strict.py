import json
import sys
from pathlib import Path

import pytest

from scripts import xxl_KDR146_run_thesis_complete_modern as mod


def test_phase0_fails_when_autoscale_missing(tmp_path, monkeypatch):
    # Point module ROOT to a temp dir to avoid touching repo outputs
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()

    autos = mod.read_autoscale_config()
    assert autos["n_samples"] is None

    ok = mod.phase_0_preflight(autos, best_sampler="tpe", smoke=False)
    assert ok is False


def test_phase0_allows_smoke_with_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()

    autos = mod.read_autoscale_config()
    assert autos["n_samples"] is None

    ok = mod.phase_0_preflight(autos, best_sampler="tpe", smoke=True)
    assert ok is True
    # smoke defaults applied
    assert autos["n_samples"] == 40
    assert autos["alpha"] == pytest.approx(0.33)
