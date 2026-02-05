from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


def _get_repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).resolve().parents[2]


def _build_constrain_bounds(
    constrain_a: tuple | None,
    constrain_b: tuple | None,
    constrain_c: tuple | None,
    constrain_min_dist: tuple | None,
) -> dict | None:
    if not any([constrain_a, constrain_b, constrain_c, constrain_min_dist]):
        return None
    return {
        "a_min": constrain_a[0] if constrain_a else None,
        "a_max": constrain_a[1] if constrain_a else None,
        "b_min": constrain_b[0] if constrain_b else None,
        "b_max": constrain_b[1] if constrain_b else None,
        "c_min": constrain_c[0] if constrain_c else None,
        "c_max": constrain_c[1] if constrain_c else None,
        "min_dist_min": constrain_min_dist[0] if constrain_min_dist else None,
        "min_dist_max": constrain_min_dist[1] if constrain_min_dist else None,
    }


def _get_run_optuna():
    """Late import hook for run_optuna to keep tests lightweight."""
    from scripts.optuna_optimize import run_optuna

    return run_optuna


def run_single_optuna(
    sampler: str,
    seed: int,
    n_trials: int,
    n_candidates: int,
    preselection_flag: str | None,
    exp_desc: str,
    dataset: str | None = None,
    fixed_n_samples: int | None = None,
    constrain_a: tuple | None = None,
    constrain_b: tuple | None = None,
    constrain_c: tuple | None = None,
    constrain_min_dist: tuple | None = None,
) -> Dict:
    """Run Optuna optimization for one sampler/seed and return run metadata.

    Direct call (no subprocess) to scripts.optuna_optimize.run_optuna().
    """
    run_optuna = _get_run_optuna()

    ROOT = _get_repo_root()

    dataset_prefix = f"{dataset}_" if dataset else ""
    exp_name = f"{dataset_prefix}{sampler}_{n_trials}trials_s{seed}"

    constrain_bounds = _build_constrain_bounds(
        constrain_a, constrain_b, constrain_c, constrain_min_dist
    )

    if fixed_n_samples is not None:
        n_samples = int(fixed_n_samples)
        n_samples_range = None
    else:
        n_samples = 34
        n_samples_range = (30, 50)

    # Run optimization directly
    run_optuna(
        n_trials=n_trials,
        n_candidates=n_candidates,
        n_samples=n_samples,
        n_samples_range=n_samples_range,
        seed=seed,
        sampler_name=sampler,
        exp_name=exp_name,
        out_dir=ROOT / "outputs",
        constrain_bounds=constrain_bounds,
    )

    # Find latest run dir
    run_dirs = sorted((ROOT / "outputs" / "runs").glob(f"*{exp_name}"))
    if not run_dirs:
        msg = f"No run dir found for {exp_name}"
        raise FileNotFoundError(msg)
    run_dir = run_dirs[-1]
    trials_csv = run_dir / "results" / "trials.csv"

    # Retry loop for output integrity (filesystem latency)
    for _ in range(5):
        if trials_csv.exists() and trials_csv.stat().st_size > 0:
            break
        time.sleep(1)

    if not trials_csv.exists():
        raise FileNotFoundError(f"trials.csv missing in {run_dir}")

    df = pd.read_csv(trials_csv)
    df = df[df["value"].notna()]

    # Normalize trial number column names across different optuna versions/formats
    for col in ("trial_number", "number", "trial"):
        if col in df.columns:
            df = df.rename(columns={col: "trial_number"})
            break

    best_val = float(df["value"].max()) if len(df) > 0 else float("nan")
    if len(df) > 0 and "trial_number" in df.columns:
        best_trial = int(df.loc[df["value"].idxmax(), "trial_number"])
    else:
        best_trial = -1
    cumulative_best = (
        df["value"].expanding().max() if len(df) > 0 else pd.Series(dtype=float)
    )
    threshold_idx = (
        (cumulative_best >= (best_val * 0.99)).idxmax()
        if (len(df) > 0 and (cumulative_best >= (best_val * 0.99)).any())
        else (len(df) - 1)
    )

    return {
        "sampler": sampler,
        "seed": seed,
        "n_trials": len(df),
        "best_value": best_val,
        "best_trial": best_trial,
        "mean_value": float(df["value"].mean()) if len(df) > 0 else float("nan"),
        "std_value": float(df["value"].std()) if len(df) > 0 else float("nan"),
        "convergence_trial": int(threshold_idx),
        "convergence_ratio": (
            float(threshold_idx / len(df)) if len(df) > 0 else float("nan")
        ),
        "run_dir": str(run_dir),
        "exp_desc": exp_desc,
        "preselection_flag": preselection_flag,
    }


def compare_and_analyze(results_df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw
    raw_csv = out_dir / "per_run_results.csv"
    results_df.to_csv(raw_csv, index=False)
    print(f"Saved per-run results: {raw_csv}")

    # Group by sampler
    grouped = results_df.groupby("sampler")["best_value"].apply(list).to_dict()

    # Basic summary
    summary = (
        results_df.groupby("sampler")["best_value"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    # Bootstrap 95% CI for mean & median per sampler
    rng = np.random.default_rng(42)
    ci_rows = []
    for sampler, vals in grouped.items():
        arr = np.array(vals)
        boot_means = []
        boot_medians = []
        for _ in range(2000):
            resample = rng.choice(arr, size=arr.size, replace=True)
            boot_means.append(resample.mean())
            boot_medians.append(np.median(resample))
        mean_lo, mean_hi = np.percentile(boot_means, [2.5, 97.5])
        med_lo, med_hi = np.percentile(boot_medians, [2.5, 97.5])
        ci_rows.append(
            {
                "sampler": sampler,
                "mean_ci_lo": mean_lo,
                "mean_ci_hi": mean_hi,
                "median_ci_lo": med_lo,
                "median_ci_hi": med_hi,
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on="sampler")
    summary_file = out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

    # Statistical tests: pairwise Mann-Whitney U
    stats = []
    samplers = list(grouped.keys())
    for i in range(len(samplers)):
        for j in range(i + 1, len(samplers)):
            s1 = samplers[i]
            s2 = samplers[j]
            a = np.array(grouped[s1])
            b = np.array(grouped[s2])
            if mannwhitneyu is None:
                pval = float("nan")
            else:
                try:
                    u = mannwhitneyu(a, b, alternative="two-sided")
                    pval = float(u.pvalue)
                except Exception:
                    pval = float("nan")
            # Cohen's d
            pooled_std = (
                np.sqrt(
                    (
                        (a.size - 1) * a.std(ddof=1) ** 2
                        + (b.size - 1) * b.std(ddof=1) ** 2
                    )
                    / (a.size + b.size - 2)
                )
                if (a.size > 1 and b.size > 1)
                else float("nan")
            )
            cohen_d = (
                (a.mean() - b.mean()) / pooled_std
                if pooled_std and not np.isnan(pooled_std) and pooled_std != 0
                else float("nan")
            )
            stats.append(
                {"sampler1": s1, "sampler2": s2, "pvalue": pval, "cohens_d": cohen_d}
            )

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "pairwise_stats.csv", index=False)
    print(f"Saved pairwise statistics: {out_dir / 'pairwise_stats.csv'}")

    # Plots: boxplot of best values
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [grouped[s] for s in samplers]
    ax.boxplot(data, tick_labels=[s.upper() for s in samplers], patch_artist=True)
    ax.set_title("Best Value Distribution per Sampler (across seeds)")
    ax.set_ylabel("Objective Value")

    # Annotate pairwise p-values
    y_max = max([np.max(d) for d in data])
    y_min = min([np.min(d) for d in data])
    y = y_max + (y_max - y_min) * 0.05
    for idx, row in stats_df.iterrows() if "stats_df" in locals() else []:
        p = row["pvalue"]
        if not np.isnan(p):
            txt = f"p={p:.3f}"
            ax.text(
                0.5,
                y + idx * (y_max - y_min) * 0.02,
                f"{row['sampler1'].upper()} vs {row['sampler2'].upper()}: {txt}",
                fontsize=8,
            )

    plt.tight_layout()
    bp = out_dir / "best_value_boxplot.png"
    plt.savefig(bp, dpi=300)
    plt.close()
    print(f"Saved boxplot: {bp}")

    # Convergence summary: compile median cumulative best across seeds for each sampler
    median_curves = {}
    for sampler in samplers:
        # Collect cumulative arrays
        cumuls = []
        for _, row in results_df[results_df["sampler"] == sampler].iterrows():
            trials_csv = Path(row["run_dir"]) / "results" / "trials.csv"
            if trials_csv.exists():
                df = pd.read_csv(trials_csv)
                df = df[df["value"].notna()]
                cumuls.append(df["value"].expanding().max().values)
        if cumuls:
            maxlen = max(len(a) for a in cumuls)
            arr = np.array(
                [
                    np.pad(a, (0, maxlen - len(a)), constant_values=np.nan)
                    for a in cumuls
                ]
            )
            median_curves[sampler] = np.nanmedian(arr, axis=0)

    conv_file = ""
    if median_curves:
        fig, ax = plt.subplots(figsize=(10, 6))
        for s, curve in median_curves.items():
            ax.plot(curve, label=s.upper())
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Median Best Objective")
        ax.set_title("Median Convergence Curves (across seeds)")
        ax.legend()
        plt.tight_layout()
        conv_file = out_dir / "median_convergence.png"
        plt.savefig(conv_file, dpi=300)
        plt.close()
        print(f"Saved convergence plot: {conv_file}")

    return {
        "summary_file": str(summary_file),
        "pairwise_stats": str(out_dir / "pairwise_stats.csv"),
        "plots": [str(bp), str(conv_file) if median_curves else ""],
    }


def compare_multi_seed(
    samplers: Iterable[str],
    seeds: Iterable[int],
    n_trials: int,
    n_candidates: Optional[int] = None,
    datasets: Optional[list[str]] = None,
    output: Optional[str] = None,
    n_samples: Optional[int] = None,
    constrain_a_min: Optional[float] = None,
    constrain_a_max: Optional[float] = None,
    constrain_b_min: Optional[float] = None,
    constrain_b_max: Optional[float] = None,
    constrain_c_min: Optional[float] = None,
    constrain_c_max: Optional[float] = None,
    constrain_min_dist_min: Optional[int] = None,
    constrain_min_dist_max: Optional[int] = None,
) -> dict:
    ROOT = _get_repo_root()
    csv_meta_path = ROOT / "data" / "new_all_tiles.csv"

    datasets = datasets or ["full"]

    timestamp = datetime.now().strftime("%Y%m%d_T%H%M%S")
    if output:
        global_out_dir = Path(output)
    else:
        global_out_dir = ROOT / "outputs" / "runs" / f"sampler_multi_{timestamp}"
    global_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset in datasets:
        print(f"\n=== Running dataset: {dataset} ===\n")
        preselection_flag = "--hamburg" if dataset == "hamburg" else None

        if n_candidates is None:
            try:
                df_meta = pd.read_csv(csv_meta_path)
                n_candidates_local = len(df_meta)
                print(f"[INFO] Auto-detected n_candidates from CSV: {n_candidates_local}")
            except FileNotFoundError:
                n_candidates_local = 676
                print(f"WARNING: {csv_meta_path} not found, using default {n_candidates_local}")
        else:
            n_candidates_local = n_candidates

        dataset_out = global_out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        for sampler in samplers:
            for seed in seeds:
                print(f"Starting run: dataset={dataset} sampler={sampler} seed={seed}")

                constrain_a = (
                    (constrain_a_min, constrain_a_max)
                    if constrain_a_min is not None
                    else None
                )
                constrain_b = (
                    (constrain_b_min, constrain_b_max)
                    if constrain_b_min is not None
                    else None
                )
                constrain_c = (
                    (constrain_c_min, constrain_c_max)
                    if constrain_c_min is not None
                    else None
                )
                constrain_min_dist = (
                    (constrain_min_dist_min, constrain_min_dist_max)
                    if constrain_min_dist_min is not None
                    else None
                )

                meta = run_single_optuna(
                    sampler,
                    seed,
                    n_trials,
                    n_candidates_local,
                    preselection_flag,
                    f"Multi-seed comparison ({dataset})",
                    dataset=dataset,
                    fixed_n_samples=n_samples,
                    constrain_a=constrain_a,
                    constrain_b=constrain_b,
                    constrain_c=constrain_c,
                    constrain_min_dist=constrain_min_dist,
                )
                meta["dataset"] = dataset
                all_results.append(meta)

    df_results = pd.DataFrame(all_results)

    analysis = compare_and_analyze(df_results, global_out_dir)
    print("Analysis complete:", analysis)

    # Determine best sampler (mean best_value across seeds)
    try:
        best_sampler = df_results.groupby("sampler")["best_value"].mean().idxmax()
        best_score = float(df_results.groupby("sampler")["best_value"].mean().max())
    except Exception:
        best_sampler = None
        best_score = None

    selected = {
        "best": best_sampler,
        "metric": "mean_best_value",
        "score": best_score,
        "n_trials": int(n_trials),
        "seeds": list(seeds),
        "datasets": datasets,
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(global_out_dir),
    }

    # Persist selected sampler artifact to a canonical location for the monitor
    try:
        sel_file = ROOT / "outputs" / "selected_sampler.json"
        sel_file.write_text(json.dumps(selected, indent=2))
        print(f"Wrote selected sampler artifact: {sel_file}")
    except Exception as e:
        print(f"Warning: could not write selected_sampler.json: {e}")

    # Also write inside the experiment-specific output folder for convenience
    try:
        (global_out_dir / "selected_sampler.json").write_text(
            json.dumps(selected, indent=2)
        )
    except Exception:
        pass

    return {
        "results": df_results,
        "analysis": analysis,
        "output_dir": str(global_out_dir),
    }


def compare_seeded_vs_unseeded(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    ROOT = _get_repo_root()
    cfg_path = config_path or (ROOT / "config" / "pipeline_config.yaml")

    import yaml

    cfg = yaml.safe_load(open(cfg_path))

    n_samples = cfg.get("selection", {}).get("n_samples", 34)
    alpha = cfg.get("selection", {}).get("alpha_visual", 0.7)
    beta = cfg.get("selection", {}).get("beta_spatial", 0.05)
    gamma = cfg.get("selection", {}).get("gamma_temporal", 0.25)
    min_distance_km = cfg.get("selection", {}).get("min_distance_km", 40.0)
    batch_size = cfg.get("feature_extraction", {}).get("batch_size", 8)

    OUT = output_dir or (ROOT / "outputs" / "seed_benchmark")
    OUT.mkdir(parents=True, exist_ok=True)

    # Runtime imports to avoid heavy deps at import-time
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.analysis.metrics import compute_metrics

    # Load cached features & metadata
    features = load_or_extract_features(
        OUT,
        csv_meta=str(ROOT / "data" / "new_all_tiles.csv"),
        batch_size=batch_size,
        cache=True,
    )
    metadata = load_metadata(str(ROOT / "data" / "new_all_tiles.csv"))

    # compute cluster labels using existing pipeline to be consistent
    n_clusters_cfg = cfg.get("clustering", {}).get("n_clusters", 8)
    clustering = ClusteringPipeline(n_clusters=n_clusters_cfg)
    _, cluster_labels = clustering.fit_transform(features)

    results = []

    # Two scenarios: baseline (no seed) and seeded (Hamburg)
    scenarios = [
        ("no_seed", None, None),
        ("seed_Hamburg_name", ["Hamburg"], None),
    ]

    for tag, pre_names, pre_idxs in scenarios:
        ds = DiversitySelector(
            n_samples=n_samples, use_multi_criteria=True, random_state=42
        )
        selected = ds.select(
            features=features,
            metadata=metadata,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
            spatial_constraint=True,
            min_distance_km=min_distance_km,
            pre_selected=pre_idxs,
            pre_selected_names=pre_names,
        )

        metrics = compute_metrics(selected, metadata, cluster_labels, features)
        metrics.update(
            {
                "scenario": tag,
                "pre_selected_names": pre_names,
                "pre_selected_indices": pre_idxs,
                "n_selected": len(selected),
            }
        )

        # Also save the selection CSV snapshot
        sel_df = metadata.iloc[selected].copy()
        sel_df["selection_rank"] = range(len(sel_df))
        sel_out = OUT / f"selection_{tag}.csv"
        sel_df.to_csv(sel_out, index=False)

        results.append(metrics)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUT / "seed_vs_unseed_metrics.csv", index=False)

    # Write small Markdown summary
    md = OUT / "seed_vs_unseed_report.md"
    with open(md, "w") as f:
        f.write("# Seed vs No-Seed Selection Benchmark\n\n")
        f.write(
            "This short report compares baseline selection and selection seeded with 'Hamburg'.\n\n"
        )
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write("\n" + df.to_string(index=False) + "\n")
        f.write("\n\nSelections saved in this folder for inspection.\n")

    print("Done. Results:", OUT)
    return OUT


def benchmark_seed(
    seeds: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
    subset_n: int = 200,
) -> Path:
    ROOT = _get_repo_root()
    OUT = output_dir or (ROOT / "outputs")
    OUT.mkdir(exist_ok=True, parents=True)

    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.clustering import ClusteringPipeline

    # Ensure features/metadata exists or extract on-the-fly
    csv_meta = OUT / "metadata.csv"
    csv_meta = str(csv_meta) if csv_meta.exists() else None
    features = load_or_extract_features(
        out_dir=OUT, csv_meta=csv_meta, batch_size=16, cache=True
    )
    _metadata = load_metadata(
        csv_meta if csv_meta is not None else str(ROOT / "data" / "new_all_tiles.csv")
    )

    # subset size for quick timing
    subset_n = min(subset_n, len(features))
    feat_sub = features[:subset_n]

    results = []

    print("Testing utopian (non-deterministic, n_jobs=-1) setting...")
    try:
        t0 = time.perf_counter()
        cl = ClusteringPipeline(n_clusters=8, umap_random_state=None, umap_n_jobs=-1)
        _emb = cl.fit_transform(feat_sub)[0]
        t = time.perf_counter() - t0
        print(f"UTOPIAN success: {t:.3f}s")
        results.append({"mode": "utopian", "seed": None, "n_jobs": -1, "time_s": t})
    except Exception as e:
        print("UTOPIAN failed:", e)

    seed_list = seeds or [42, 0, 1, 123, 999, 2026]
    for s in seed_list:
        print(f"Testing seed={s} (deterministic, single-thread) ...")
        try:
            t0 = time.perf_counter()
            cl = ClusteringPipeline(
                n_clusters=8, umap_random_state=int(s), umap_n_jobs=1
            )
            _emb = cl.fit_transform(feat_sub)[0]
            t = time.perf_counter() - t0
            print(f"  seed {s} success: {t:.3f}s")
            results.append({"mode": "seeded", "seed": int(s), "n_jobs": 1, "time_s": t})
        except Exception as e:
            print(f"  seed {s} failed:", e)

    # Save results
    out_csv = OUT / "seed_benchmark_results.csv"
    keys = ["mode", "seed", "n_jobs", "time_s"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\nBenchmark results saved to", out_csv)

    # Decide recommended configuration: fastest overall
    if results:
        best = min(results, key=lambda x: x["time_s"])
        print("\nBest config:")
        print(best)
        rec_file = OUT / "seed_benchmark_recommendation.txt"
        with open(rec_file, "w") as f:
            f.write("Best config (fastest):\n")
            f.write(str(best) + "\n")
        print("Recommendation written to", rec_file)
    else:
        print("No successful runs recorded.")

    return out_csv


@cli_command(
    "compare-samplers",
    help="Compare samplers across multiple seeds and datasets",
    args={
        "samplers": {
            "nargs": "+",
            "default": ["qmc", "tpe", "cmaes"],
            "help": "Samplers to compare",
        },
        "seeds": {
            "nargs": "+",
            "type": int,
            "default": [42, 43, 44, 45, 46],
            "help": "Random seeds to use",
        },
        "n_trials": {
            "type": int,
            "default": 500,
            "help": "Number of trials per run",
        },
        "n_candidates": {
            "type": int,
            "default": None,
            "help": "Number of candidates",
        },
        "datasets": {
            "nargs": "+",
            "choices": ["hamburg", "kdr100", "full"],
            "default": None,
            "help": "Datasets to run on",
        },
        "sequential": {
            "action": "store_true",
            "help": "Run sequentially (default)",
        },
        "output": {
            "type": str,
            "default": None,
            "help": "Output directory",
        },
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Fixed n_samples for all runs",
        },
        "constrain_a_min": {
            "type": float,
            "default": None,
            "help": "Constrain alpha (a) lower bound",
        },
        "constrain_a_max": {
            "type": float,
            "default": None,
            "help": "Constrain alpha (a) upper bound",
        },
        "constrain_b_min": {
            "type": float,
            "default": None,
            "help": "Constrain beta (b) lower bound",
        },
        "constrain_b_max": {
            "type": float,
            "default": None,
            "help": "Constrain beta (b) upper bound",
        },
        "constrain_c_min": {
            "type": float,
            "default": None,
            "help": "Constrain gamma (c) lower bound",
        },
        "constrain_c_max": {
            "type": float,
            "default": None,
            "help": "Constrain gamma (c) upper bound",
        },
        "constrain_min_dist_min": {
            "type": int,
            "default": None,
            "help": "Constrain min_distance lower bound",
        },
        "constrain_min_dist_max": {
            "type": int,
            "default": None,
            "help": "Constrain min_distance upper bound",
        },
    },
)
def main(
    samplers: list[str] = None,
    seeds: list[int] = None,
    n_trials: int = 500,
    n_candidates: Optional[int] = None,
    datasets: Optional[list[str]] = None,
    sequential: bool = False,
    output: Optional[str] = None,
    n_samples: Optional[int] = None,
    constrain_a_min: Optional[float] = None,
    constrain_a_max: Optional[float] = None,
    constrain_b_min: Optional[float] = None,
    constrain_b_max: Optional[float] = None,
    constrain_c_min: Optional[float] = None,
    constrain_c_max: Optional[float] = None,
    constrain_min_dist_min: Optional[int] = None,
    constrain_min_dist_max: Optional[int] = None,
) -> int:
    """CLI entry point for compare samplers (multi-seed mode)."""
    # Set defaults if None
    if samplers is None:
        samplers = ["qmc", "tpe", "cmaes"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    
    # Call the comparison
    compare_multi_seed(
        samplers=samplers,
        seeds=seeds,
        n_trials=n_trials,
        n_candidates=n_candidates,
        datasets=datasets,
        output=output,
        n_samples=n_samples,
        constrain_a_min=constrain_a_min,
        constrain_a_max=constrain_a_max,
        constrain_b_min=constrain_b_min,
        constrain_b_max=constrain_b_max,
        constrain_c_min=constrain_c_min,
        constrain_c_max=constrain_c_max,
        constrain_min_dist_min=constrain_min_dist_min,
        constrain_min_dist_max=constrain_min_dist_max,
    )
    return 0



if __name__ == "__main__":
    # Use CLI: dataselector compare-samplers --samplers X --seeds Y --n-trials Z
    raise SystemExit(1)

