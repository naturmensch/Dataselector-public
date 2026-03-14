"""Tests for dataselector.workflows.compare_samplers module."""

import numpy as np
import pandas as pd
import pytest
import yaml


def _write_minimal_metadata_csv(repo_root):
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "SheetNumber": [1],
            "shortName": ["S1"],
            "longName": ["tile_1"],
            "city": ["test_city"],
            "year": [1910],
            "filename": ["tile_1.tif"],
            "image_path": ["tile_1.png"],
            "ul_x": [500000.0],
            "ul_y": [5900000.0],
            "lr_x": [500100.0],
            "lr_y": [5899900.0],
            "width_px": [1000],
            "height_px": [1000],
            "pixel_width": [0.1],
            "pixel_height": [-0.1],
            "data_quality": ["ok"],
        }
    ).to_csv(data_dir / "new_all_tiles.csv", index=False)


def test_run_single_optuna_missing_run_dir(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    # Patch repo root to tmp_path
    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)
    _write_minimal_metadata_csv(tmp_path)

    monkeypatch.setattr(
        compare_samplers, "_get_run_optuna", lambda: (lambda **kwargs: None)
    )

    with pytest.raises(FileNotFoundError, match="No run dir found"):
        compare_samplers.run_single_optuna(
            sampler="cmaes",
            seed=42,
            n_trials=10,
            n_candidates=100,
            preselection_flag=None,
            exp_desc="desc",
            dataset="hamburg",
        )


def test_run_single_optuna_success(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)
    _write_minimal_metadata_csv(tmp_path)

    monkeypatch.setattr(
        compare_samplers, "_get_run_optuna", lambda: (lambda **kwargs: None)
    )

    exp_name = "hamburg_cmaes_10trials_s42"
    run_dir = tmp_path / "outputs" / "runs" / exp_name
    (run_dir / "results").mkdir(parents=True)

    df = pd.DataFrame({"trial_number": range(10), "value": range(10)})
    df.to_csv(run_dir / "results" / "trials.csv", index=False)

    res = compare_samplers.run_single_optuna(
        sampler="cmaes",
        seed=42,
        n_trials=10,
        n_candidates=100,
        preselection_flag=None,
        exp_desc="desc",
        dataset="hamburg",
    )

    assert res["n_trials"] == 10
    assert res["best_value"] == 9.0
    assert res["run_dir"] == str(run_dir)


def test_compare_multi_seed_creates_summary(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)

    # Prepare a fake run_dir with trials.csv
    run_dir = tmp_path / "outputs" / "runs" / "full_qmc_1trials_s1"
    (run_dir / "results").mkdir(parents=True)
    df = pd.DataFrame({"trial_number": [0, 1], "value": [0.1, 0.2]})
    df.to_csv(run_dir / "results" / "trials.csv", index=False)

    def fake_run_single_optuna(*args, **kwargs):
        return {
            "sampler": "qmc",
            "seed": 1,
            "n_trials": 2,
            "best_value": 0.2,
            "best_trial": 1,
            "mean_value": 0.15,
            "std_value": 0.05,
            "convergence_trial": 1,
            "convergence_ratio": 0.5,
            "run_dir": str(run_dir),
            "exp_desc": "desc",
            "preselection_flag": None,
        }

    monkeypatch.setattr(compare_samplers, "run_single_optuna", fake_run_single_optuna)

    out_dir = tmp_path / "outputs" / "runs" / "sampler_test"
    result = compare_samplers.compare_multi_seed(
        samplers=["qmc"],
        seeds=[1],
        n_trials=1,
        n_candidates=10,
        datasets=["full"],
        output=str(out_dir),
    )

    summary = out_dir / "summary.csv"
    assert summary.exists()
    assert isinstance(result, dict)


def _install_seed_vs_unseed_stubs(monkeypatch, tmp_path):
    from dataselector.analysis import metrics as metrics_mod
    from dataselector.data import io as data_io
    from dataselector.data import metadata_source, spatial_schema
    from dataselector.selection import clustering, diversity_selector
    from dataselector.workflows import compare_samplers, objective_scoring

    features = np.arange(60, dtype=float).reshape(20, 3)
    metadata = pd.DataFrame(
        {
            "year": np.linspace(1880, 1918, 20, dtype=int),
            "city": [f"city_{i}" for i in range(20)],
            "ul_x": 500000.0 + np.arange(20) * 1000.0,
            "ul_y": 5900000.0 + np.arange(20) * 1000.0,
            "lr_x": 500500.0 + np.arange(20) * 1000.0,
            "lr_y": 5899500.0 + np.arange(20) * 1000.0,
        }
    )

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        metadata_source, "canonical_metadata_path", lambda _root: tmp_path / "data.csv"
    )
    monkeypatch.setattr(data_io, "load_or_extract_features", lambda *a, **k: features)
    monkeypatch.setattr(data_io, "load_metadata", lambda *a, **k: metadata.copy())
    monkeypatch.setattr(
        objective_scoring,
        "compute_baselines",
        lambda **kwargs: (1.0, 1.0),
    )
    monkeypatch.setattr(
        spatial_schema,
        "normalize_spatial_schema",
        lambda meta, **kwargs: meta.assign(
            center_x=(meta["ul_x"] + meta["lr_x"]) / 2.0,
            center_y=(meta["ul_y"] + meta["lr_y"]) / 2.0,
        ).copy(),
    )

    class FakeClusteringPipeline:
        def __init__(self, n_clusters, *args, **kwargs):
            self.n_clusters = int(n_clusters)

        def fit_transform(self, _features):
            labels = np.arange(len(_features), dtype=int) % self.n_clusters
            return _features[:, :2], labels

    class FakeDiversitySelector:
        def __init__(
            self, n_samples, use_multi_criteria=True, random_state=42, **kwargs
        ):
            self.n_samples = int(n_samples)
            self.random_state = int(random_state)

        def select(
            self,
            features,
            metadata,
            alpha_visual,
            beta_spatial,
            gamma_temporal,
            spatial_constraint,
            min_distance_km,
            pre_selected=None,
            pre_selected_names=None,
        ):
            rng = np.random.default_rng(self.random_state)
            picked = rng.choice(
                len(features),
                size=min(self.n_samples, len(features)),
                replace=False,
            ).astype(int)
            if pre_selected_names and 0 not in picked:
                picked[0] = 0
            return np.sort(picked)

        def _calculate_diversity_score(self, arr):
            if len(arr) == 0:
                return 0.0
            return float(np.mean(arr))

    def fake_compute_metrics(selected_idx, metadata, cluster_labels, features):
        idx = np.asarray(selected_idx, dtype=int)
        years = (
            metadata.iloc[idx]["year"].to_numpy(dtype=float)
            if len(idx)
            else np.array([])
        )
        spatial = idx.astype(float)
        return {
            "n_selected": int(len(idx)),
            "temporal_std": float(years.std()) if len(years) > 1 else 0.0,
            "temporal_range": (int(years.max() - years.min()) if len(years) > 1 else 0),
            "wwi_percent": (
                float(np.mean((years >= 1914) & (years <= 1918)) * 100.0)
                if len(years)
                else 0.0
            ),
            "spatial_mean_km": float(spatial.mean()) if len(spatial) else 0.0,
            "spatial_min_km": float(spatial.min()) if len(spatial) else 0.0,
            "clusters_covered": (
                int(len(np.unique(cluster_labels[idx]))) if len(idx) else 0
            ),
        }

    def fake_objective(
        *,
        selector,
        features,
        spatial_meta,
        selected,
        baseline_diversity,
        baseline_spread,
        target_n,
    ):
        idx = np.asarray(selected, dtype=int)
        bonus = 0.05 if 0 in idx else 0.0
        score = float(len(idx) / max(1, int(target_n)) + bonus)
        return {
            "objective_score": score,
            "objective_score_raw": score,
            "objective_diversity_norm": score,
            "objective_spread_norm": score,
            "objective_feasibility_ratio": float(len(idx) / max(1, int(target_n))),
            "objective_infeasible": len(idx) < int(target_n),
        }

    monkeypatch.setattr(clustering, "ClusteringPipeline", FakeClusteringPipeline)
    monkeypatch.setattr(diversity_selector, "DiversitySelector", FakeDiversitySelector)
    monkeypatch.setattr(metrics_mod, "compute_metrics", fake_compute_metrics)
    monkeypatch.setattr(
        compare_samplers, "_compute_objective_for_selection", fake_objective
    )


def test_compare_seeded_vs_unseeded_multi_seed_outputs(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    _install_seed_vs_unseed_stubs(monkeypatch, tmp_path)

    cfg_path = tmp_path / "pipeline_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "selection": {
                    "n_samples": 30,
                    "alpha_visual": 0.37906427571507917,
                    "beta_spatial": 0.21272076682352906,
                    "gamma_temporal": 0.40821495746139186,
                    "min_distance_km": 57.0,
                },
                "clustering": {"n_clusters": 8},
                "feature_extraction": {"batch_size": 8},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "anchor_evidence" / "isolated"
    out = compare_samplers.compare_seeded_vs_unseeded(
        config_path=cfg_path,
        output_dir=out_dir,
        seeds=[42, 43, 44, 45, 46],
    )

    assert out == out_dir
    raw_csv = out_dir / "seed_vs_unseed_metrics.csv"
    summary_csv = out_dir / "seed_vs_unseed_summary.csv"
    stats_csv = out_dir / "seed_vs_unseed_stats.csv"
    report_md = out_dir / "seed_vs_unseed_report.md"
    assert raw_csv.exists()
    assert summary_csv.exists()
    assert stats_csv.exists()
    assert report_md.exists()

    raw_df = pd.read_csv(raw_csv)
    assert len(raw_df) == 10  # 2 scenarios x 5 seeds
    assert set(raw_df["scenario"]) == {"no_seed", "seed_Hamburg_name"}
    assert set(raw_df["seed"]) == {42, 43, 44, 45, 46}

    stats_df = pd.read_csv(stats_csv)
    assert "objective_score" in set(stats_df["endpoint"])


def test_paired_stats_behavior_identical_vs_shifted():
    from dataselector.workflows import compare_samplers

    seeds = [42, 43, 44, 45, 46]
    identical_rows = []
    shifted_rows = []
    for i, seed in enumerate(seeds):
        same = 1.0 + i * 0.1
        base = 1.0 + i * 0.1
        identical_rows.append(
            {"seed": seed, "scenario": "no_seed", "objective_score": same}
        )
        identical_rows.append(
            {"seed": seed, "scenario": "seed_Hamburg_name", "objective_score": same}
        )
        shifted_rows.append(
            {"seed": seed, "scenario": "no_seed", "objective_score": base}
        )
        shifted_rows.append(
            {
                "seed": seed,
                "scenario": "seed_Hamburg_name",
                "objective_score": base + 2.0,
            }
        )

    stats_same = compare_samplers._compute_paired_endpoint_stats(
        pd.DataFrame(identical_rows),
        endpoints=["objective_score"],
    )
    row_same = stats_same.iloc[0]
    assert row_same["mean_delta"] == pytest.approx(0.0)
    assert row_same["p_value_exact_paired"] == pytest.approx(1.0)

    stats_shifted = compare_samplers._compute_paired_endpoint_stats(
        pd.DataFrame(shifted_rows),
        endpoints=["objective_score"],
    )
    row_shifted = stats_shifted.iloc[0]
    assert row_shifted["mean_delta"] > 0.0
    assert row_shifted["p_value_exact_paired"] <= 0.1


def test_cli_compare_seed_vs_unseeded_new_flags_smoke(tmp_path, monkeypatch):
    from dataselector.cli import main
    from dataselector.workflows import compare_samplers

    captured: dict[str, object] = {}

    def fake_compare_seeded_vs_unseeded(**kwargs):
        captured["kwargs"] = kwargs
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "seed_vs_unseed_metrics.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_summary.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_stats.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_report.md").write_text("# x\n", encoding="utf-8")
        return out

    def fake_production_quick(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "production_delta.csv").write_text("metric\n", encoding="utf-8")
        (out / "production_delta_report.md").write_text("# x\n", encoding="utf-8")
        captured["production"] = kwargs
        return out

    monkeypatch.setattr(
        compare_samplers,
        "compare_seeded_vs_unseeded",
        fake_compare_seeded_vs_unseeded,
    )
    monkeypatch.setattr(
        compare_samplers,
        "compare_production_runs_quick_delta",
        fake_production_quick,
    )

    out_root = tmp_path / "anchor_evidence"
    seeded_run = tmp_path / "seeded_run"
    unseeded_run = tmp_path / "unseeded_run"
    seeded_run.mkdir()
    unseeded_run.mkdir()

    rc = main(
        [
            "compare-seed-vs-unseeded",
            "--output-dir",
            str(out_root),
            "--seeds",
            "42",
            "43",
            "44",
            "45",
            "46",
            "--n-samples",
            "30",
            "--alpha-visual",
            "0.37906427571507917",
            "--beta-spatial",
            "0.21272076682352906",
            "--gamma-temporal",
            "0.40821495746139186",
            "--min-distance-km",
            "57.0",
            "--run-production-quick",
            "--production-seeded-run",
            str(seeded_run),
            "--production-unseeded-run",
            str(unseeded_run),
        ]
    )

    assert rc == 0
    assert captured["kwargs"]["seeds"] == [42, 43, 44, 45, 46]
    assert (out_root / "isolated" / "seed_vs_unseed_metrics.csv").exists()
    assert (out_root / "isolated" / "seed_vs_unseed_summary.csv").exists()
    assert (out_root / "isolated" / "seed_vs_unseed_stats.csv").exists()
    assert (out_root / "isolated" / "seed_vs_unseed_report.md").exists()
    assert (out_root / "production_quick" / "production_delta.csv").exists()
    assert (out_root / "production_quick" / "production_delta_report.md").exists()


def test_cli_compare_seed_vs_unseeded_backward_compatible(tmp_path, monkeypatch):
    from dataselector.cli import main
    from dataselector.workflows import compare_samplers

    calls = {"production_called": False}

    def fake_compare_seeded_vs_unseeded(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "seed_vs_unseed_metrics.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_summary.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_stats.csv").write_text("x\n", encoding="utf-8")
        (out / "seed_vs_unseed_report.md").write_text("# x\n", encoding="utf-8")
        return out

    def fake_production_quick(**kwargs):
        calls["production_called"] = True
        return kwargs["output_dir"]

    monkeypatch.setattr(
        compare_samplers,
        "compare_seeded_vs_unseeded",
        fake_compare_seeded_vs_unseeded,
    )
    monkeypatch.setattr(
        compare_samplers,
        "compare_production_runs_quick_delta",
        fake_production_quick,
    )

    out_root = tmp_path / "legacy_call_ok"
    rc = main(["compare-seed-vs-unseeded", "--output-dir", str(out_root)])

    assert rc == 0
    assert (out_root / "isolated" / "seed_vs_unseed_metrics.csv").exists()
    assert calls["production_called"] is False


def test_compare_seeded_vs_unseeded_marks_non_independent_seed_replay(
    tmp_path, monkeypatch
):
    from dataselector.analysis import metrics as metrics_mod
    from dataselector.data import io as data_io
    from dataselector.data import metadata_source, spatial_schema
    from dataselector.selection import clustering, diversity_selector
    from dataselector.workflows import compare_samplers, objective_scoring

    features = np.arange(60, dtype=float).reshape(20, 3)
    metadata = pd.DataFrame(
        {
            "year": np.linspace(1880, 1918, 20, dtype=int),
            "city": [f"city_{i}" for i in range(20)],
            "shortName": [f"KDR_{i:03d}" for i in range(20)],
            "ul_x": 500000.0 + np.arange(20) * 1000.0,
            "ul_y": 5900000.0 + np.arange(20) * 1000.0,
            "lr_x": 500500.0 + np.arange(20) * 1000.0,
            "lr_y": 5899500.0 + np.arange(20) * 1000.0,
        }
    )

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        metadata_source, "canonical_metadata_path", lambda _root: tmp_path / "data.csv"
    )
    monkeypatch.setattr(data_io, "load_or_extract_features", lambda *a, **k: features)
    monkeypatch.setattr(data_io, "load_metadata", lambda *a, **k: metadata.copy())
    monkeypatch.setattr(
        objective_scoring, "compute_baselines", lambda **kwargs: (1.0, 1.0)
    )
    monkeypatch.setattr(
        spatial_schema,
        "normalize_spatial_schema",
        lambda meta, **kwargs: meta.assign(
            center_x=(meta["ul_x"] + meta["lr_x"]) / 2.0,
            center_y=(meta["ul_y"] + meta["lr_y"]) / 2.0,
        ).copy(),
    )

    class FakeClusteringPipeline:
        def __init__(self, n_clusters, *args, **kwargs):
            self.n_clusters = int(n_clusters)

        def fit_transform(self, _features):
            labels = np.arange(len(_features), dtype=int) % self.n_clusters
            return _features[:, :2], labels

    class ConstantSelectionSelector:
        def __init__(self, n_samples, **kwargs):
            self.n_samples = int(n_samples)

        def select(
            self,
            features,
            metadata,
            alpha_visual,
            beta_spatial,
            gamma_temporal,
            spatial_constraint,
            min_distance_km,
            pre_selected=None,
            pre_selected_names=None,
        ):
            base = np.arange(min(self.n_samples, len(features)), dtype=int)
            if pre_selected_names:
                # Ensure seeded scenario has a different but still deterministic signature.
                base = np.roll(base, 1)
            return base

        def _calculate_diversity_score(self, arr):
            return float(np.mean(arr)) if len(arr) else 0.0

    def fake_compute_metrics(selected_idx, metadata, cluster_labels, features):
        idx = np.asarray(selected_idx, dtype=int)
        years = (
            metadata.iloc[idx]["year"].to_numpy(dtype=float)
            if len(idx)
            else np.array([])
        )
        return {
            "n_selected": int(len(idx)),
            "temporal_std": float(years.std()) if len(years) > 1 else 0.0,
            "temporal_range": int(years.max() - years.min()) if len(years) > 1 else 0,
            "wwi_percent": 0.0,
            "spatial_mean_km": float(idx.mean()) if len(idx) else 0.0,
            "spatial_min_km": float(idx.min()) if len(idx) else 0.0,
            "clusters_covered": (
                int(len(np.unique(cluster_labels[idx]))) if len(idx) else 0
            ),
        }

    monkeypatch.setattr(clustering, "ClusteringPipeline", FakeClusteringPipeline)
    monkeypatch.setattr(
        diversity_selector, "DiversitySelector", ConstantSelectionSelector
    )
    monkeypatch.setattr(metrics_mod, "compute_metrics", fake_compute_metrics)

    out_dir = tmp_path / "anchor_evidence" / "isolated"
    compare_samplers.compare_seeded_vs_unseeded(
        output_dir=out_dir,
        seeds=[42, 43, 44, 45, 46],
        n_samples=10,
        alpha_visual=0.4,
        beta_spatial=0.3,
        gamma_temporal=0.3,
        min_distance_km=28.5,
    )

    raw_df = pd.read_csv(out_dir / "seed_vs_unseed_metrics.csv")
    assert int(raw_df["effective_replicates_no_seed"].iloc[0]) == 1
    assert int(raw_df["effective_replicates_seed_Hamburg_name"].iloc[0]) == 1
    assert raw_df["inference_status"].iloc[0] == "non_independent_seed_replay"

    stats_df = pd.read_csv(out_dir / "seed_vs_unseed_stats.csv")
    assert (stats_df["inference_status"] == "non_independent_seed_replay").all()
    assert stats_df["p_value_exact_paired"].isna().all()
