from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataselector.workflows.bootstrap as bootstrap_mod
from dataselector.workflows.bootstrap import run_bootstrap_pareto

pytestmark = pytest.mark.integration


def _prepare_common_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pareto = pd.DataFrame(
        {
            "alpha": [0.6],
            "beta": [0.2],
            "gamma": [0.2],
            "min_distance_km": [28],
            "n_selected": [10],
        }
    )
    pareto_csv = tmp_path / "pareto.csv"
    out_csv = tmp_path / "bootstrap.csv"
    pareto.to_csv(pareto_csv, index=False)
    return pareto_csv, out_csv


def _patch_lightweight_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "ul_x": [500000, 500100, 500200, 500300, 500400, 500500],
            "ul_y": [5900000, 5900100, 5900200, 5900300, 5900400, 5900500],
            "lr_x": [500050, 500150, 500250, 500350, 500450, 500550],
            "lr_y": [5899950, 5900050, 5900150, 5900250, 5900350, 5900450],
            "year": [1900, 1901, 1902, 1903, 1904, 1905],
            "tile_name": ["A", "B", "C", "D", "E", "F"],
        }
    )
    features = (
        np.random.default_rng(7).normal(size=(len(metadata), 4)).astype("float32")
    )

    monkeypatch.setattr(bootstrap_mod, "_get_repo_root", lambda: tmp_path)

    import dataselector.data.io as io_mod
    import dataselector.selection.clustering as clustering_mod
    import dataselector.selection.diversity_selector as selector_mod

    monkeypatch.setattr(io_mod, "load_metadata", lambda *_a, **_k: metadata.copy())
    monkeypatch.setattr(
        io_mod, "load_or_extract_features", lambda *_a, **_k: features.copy()
    )

    class _DummyClustering:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, feat):
            return feat[:, :2], np.zeros(len(feat), dtype=int)

    class _DummySelector:
        def __init__(self, *args, **kwargs):
            pass

        def select(self, features, metadata, **kwargs):
            return np.arange(min(4, len(features)))

    monkeypatch.setattr(clustering_mod, "ClusteringPipeline", _DummyClustering)
    monkeypatch.setattr(selector_mod, "DiversitySelector", _DummySelector)


def test_bootstrap_pareto_ensemble_mode_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_lightweight_runtime(monkeypatch, tmp_path)
    pareto_csv, out_csv = _prepare_common_inputs(tmp_path)

    def _fake_bootstrap_candidate(*_args, **_kwargs):
        n = 12
        return pd.DataFrame(
            {
                "temporal_std": np.linspace(10.0, 11.1, n),
                "spatial_mean_km": np.linspace(20.0, 25.5, n),
                "wwi_percent": np.linspace(30.0, 41.0, n),
                "clusters_covered": np.linspace(3.0, 6.0, n),
                "n_selected": np.linspace(9.0, 11.0, n),
                "jaccard_with_original": np.linspace(0.2, 0.8, n),
            }
        )

    fake_uq = types.ModuleType("dataselector.workflows.uncertainty_quantification")
    fake_uq.fit_ensemble_on_bootstrap_df = lambda **_k: ["m1", "m2", "m3"]
    fake_uq.predict_with_uncertainty = lambda _models, X_query: (
        np.full(len(X_query), 0.55),
        np.linspace(0.01, 0.08, len(X_query)),
    )

    monkeypatch.setattr(bootstrap_mod, "bootstrap_candidate", _fake_bootstrap_candidate)
    monkeypatch.setitem(
        sys.modules, "dataselector.workflows.uncertainty_quantification", fake_uq
    )

    rc = run_bootstrap_pareto(
        pareto_csv=pareto_csv,
        n_boot=12,
        output_csv=out_csv,
        random_seed=1,
        uq_method="ensemble",
    )

    assert rc == 0
    assert out_csv.exists()
    summary_path = out_csv.with_name(f"{out_csv.stem}_summary.csv")
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "method"] == "ensemble"
    assert float(summary.loc[0, "ensemble_models"]) >= 3
    assert float(summary.loc[0, "ensemble_uncertainty_mean"]) > 0.0


def test_bootstrap_pareto_invalid_uq_method_returns_code_2(tmp_path: Path):
    pareto_csv, out_csv = _prepare_common_inputs(tmp_path)

    rc = run_bootstrap_pareto(
        pareto_csv=pareto_csv,
        n_boot=12,
        output_csv=out_csv,
        random_seed=1,
        uq_method="does-not-exist",
    )

    assert rc == 2
    assert not out_csv.exists()


def test_bootstrap_pareto_ensemble_requires_sufficient_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_lightweight_runtime(monkeypatch, tmp_path)
    pareto_csv, out_csv = _prepare_common_inputs(tmp_path)

    monkeypatch.setattr(
        bootstrap_mod,
        "bootstrap_candidate",
        lambda *_a, **_k: pd.DataFrame(
            {
                "temporal_std": [1.0, 1.2],
                "spatial_mean_km": [2.0, 2.2],
                "wwi_percent": [3.0, 3.1],
                "clusters_covered": [4.0, 4.0],
                "n_selected": [10.0, 10.0],
                "jaccard_with_original": [0.4, 0.5],
            }
        ),
    )
    fake_uq = types.ModuleType("dataselector.workflows.uncertainty_quantification")
    fake_uq.fit_ensemble_on_bootstrap_df = lambda **_k: []
    fake_uq.predict_with_uncertainty = lambda *_a, **_k: (np.array([]), np.array([]))
    monkeypatch.setitem(
        sys.modules, "dataselector.workflows.uncertainty_quantification", fake_uq
    )

    rc = run_bootstrap_pareto(
        pareto_csv=pareto_csv,
        n_boot=2,
        output_csv=out_csv,
        random_seed=1,
        uq_method="ensemble",
    )

    assert rc == 2
    assert not out_csv.exists()
