import json
import sys
from pathlib import Path

import pytest

import dataselector.workflows.xxl as mod


def test_phase0_fails_when_autoscale_missing(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    autos = mod.read_autoscale_config(output_dir)
    assert autos["n_samples"] is None

    ok = mod.phase_0_preflight(autos, best_sampler="tpe", smoke=False)
    assert ok is False


def test_phase0_allows_smoke_with_defaults(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    autos = mod.read_autoscale_config(output_dir)
    assert autos["n_samples"] is None

    ok = mod.phase_0_preflight(autos, best_sampler="tpe", smoke=True)
    assert ok is True
    # smoke defaults applied
    assert autos["n_samples"] == 40
    assert autos["alpha"] == pytest.approx(0.33)
