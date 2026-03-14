from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

DEFAULT_SEED_PANEL = [42, 43, 44, 45, 46]
PRIMARY_ENDPOINT = "objective_score"
SECONDARY_ENDPOINTS = [
    "n_selected",
    "clusters_covered",
    "spatial_mean_km",
    "spatial_min_km",
    "temporal_std",
    "wwi_percent",
]
ALL_ENDPOINTS = [PRIMARY_ENDPOINT, *SECONDARY_ENDPOINTS]


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
    from dataselector.workflows.optuna_optimize import run_optuna

    return run_optuna


def _resolve_seed_panel(seeds: Optional[Iterable[int]]) -> list[int]:
    if seeds is None:
        return list(DEFAULT_SEED_PANEL)
    out: list[int] = []
    for seed in seeds:
        val = int(seed)
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError("seeds must contain at least one value.")
    return out


def _paired_standardized_delta(delta: np.ndarray) -> float:
    arr = np.asarray(delta, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean_delta = float(arr.mean())
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        if abs(mean_delta) <= 1e-12:
            return 0.0
        return float(np.sign(mean_delta) * np.inf)
    return float(mean_delta / sd)


def _exact_sign_test_pvalue(delta: np.ndarray) -> float:
    arr = np.asarray(delta, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) > 1e-12]
    n = int(arr.size)
    if n == 0:
        return 1.0
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p = min(1.0, (2.0 * tail) / (2**n))
    return float(p)


def _exact_paired_sign_flip_pvalue(delta: np.ndarray) -> float:
    arr = np.asarray(delta, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan")
    if n > 20:
        # Exact sign-flip grows as 2^n; use exact sign-test fallback for larger n.
        return _exact_sign_test_pvalue(arr)

    obs = abs(float(arr.mean()))
    n_perm = 1 << n
    count = 0
    indices = np.arange(n, dtype=int)
    for mask in range(n_perm):
        signs = np.where(((mask >> indices) & 1) == 1, -1.0, 1.0)
        perm_mean = abs(float((arr * signs).mean()))
        if perm_mean >= (obs - 1e-12):
            count += 1
    return float(count / n_perm)


def _bootstrap_mean_delta_ci(
    delta: np.ndarray, n_bootstrap: int = 5000, seed: int = 42
) -> tuple[float, float]:
    arr = np.asarray(delta, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _selection_signature(indices: Iterable[int]) -> str:
    vals = sorted(set(int(i) for i in indices))
    return "|".join(str(v) for v in vals)


def _selection_overlap_metrics(
    selected_a: Iterable[int],
    selected_b: Iterable[int],
) -> dict[str, float | int]:
    a = set(int(i) for i in selected_a)
    b = set(int(i) for i in selected_b)
    overlap = int(len(a & b))
    union = int(len(a | b))
    jaccard = float(overlap / union) if union > 0 else 1.0
    swap_count = int(max(len(a), len(b)) - overlap)
    return {
        "overlap_count": overlap,
        "swap_count": swap_count,
        "selection_jaccard": jaccard,
    }


def _compute_paired_endpoint_stats(
    df_metrics: pd.DataFrame,
    *,
    endpoints: Sequence[str],
    seeded_scenario: str = "seed_Hamburg_name",
    unseeded_scenario: str = "no_seed",
    inference_status: str = "independent",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "seed" not in df_metrics.columns or "scenario" not in df_metrics.columns:
        return pd.DataFrame()

    for endpoint in endpoints:
        if endpoint not in df_metrics.columns:
            continue
        pivot = (
            df_metrics[["seed", "scenario", endpoint]]
            .pivot_table(
                index="seed", columns="scenario", values=endpoint, aggfunc="first"
            )
            .copy()
        )
        if (
            seeded_scenario not in pivot.columns
            or unseeded_scenario not in pivot.columns
        ):
            continue
        paired = pivot[[seeded_scenario, unseeded_scenario]].dropna()
        if paired.empty:
            continue
        delta = pd.to_numeric(paired[seeded_scenario], errors="coerce") - pd.to_numeric(
            paired[unseeded_scenario], errors="coerce"
        )
        delta_arr = delta.to_numpy(dtype=float)
        delta_arr = delta_arr[np.isfinite(delta_arr)]
        if delta_arr.size == 0:
            continue
        ci_lo, ci_hi = _bootstrap_mean_delta_ci(delta_arr)
        inferential_ok = inference_status == "independent"
        rows.append(
            {
                "endpoint": endpoint,
                "is_primary_endpoint": endpoint == PRIMARY_ENDPOINT,
                "n_pairs": int(delta_arr.size),
                "mean_delta": float(delta_arr.mean()),
                "median_delta": float(np.median(delta_arr)),
                "std_delta": (
                    float(delta_arr.std(ddof=1)) if delta_arr.size > 1 else 0.0
                ),
                "ci95_mean_delta_lo": ci_lo,
                "ci95_mean_delta_hi": ci_hi,
                "p_value_exact_paired": (
                    _exact_paired_sign_flip_pvalue(delta_arr)
                    if inferential_ok
                    else float("nan")
                ),
                "effect_size_paired_std_delta": (
                    _paired_standardized_delta(delta_arr)
                    if inferential_ok
                    else float("nan")
                ),
                "delta_direction": (
                    "seeded>unseeded"
                    if float(delta_arr.mean()) > 0
                    else (
                        "seeded<unseeded"
                        if float(delta_arr.mean()) < 0
                        else "no_change"
                    )
                ),
                "inference_status": inference_status,
                "inferential_statistics_valid": bool(inferential_ok),
            }
        )
    return pd.DataFrame(rows)


def _summarize_endpoints_by_scenario(
    df_metrics: pd.DataFrame, *, endpoints: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "scenario" not in df_metrics.columns:
        return pd.DataFrame()
    for scenario in sorted(df_metrics["scenario"].dropna().unique()):
        sub = df_metrics[df_metrics["scenario"] == scenario]
        for endpoint in endpoints:
            if endpoint not in sub.columns:
                continue
            vals = pd.to_numeric(sub[endpoint], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "scenario": str(scenario),
                    "endpoint": endpoint,
                    "is_primary_endpoint": endpoint == PRIMARY_ENDPOINT,
                    "count": int(vals.size),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            )
    return pd.DataFrame(rows)


def _compute_objective_for_selection(
    *,
    selector: Any,
    features: np.ndarray,
    spatial_meta: pd.DataFrame,
    selected: np.ndarray,
    baseline_diversity: float,
    baseline_spread: float,
    target_n: int,
) -> dict[str, float | bool]:
    from dataselector.data.spatial_schema import (
        spatial_spread as compute_spatial_spread,
    )
    from dataselector.workflows.objective_scoring import normalized_objective

    selected_idx = np.asarray(selected, dtype=int)
    if selected_idx.size == 0:
        diversity = 0.0
        spread = 0.0
    else:
        diversity = float(selector._calculate_diversity_score(features[selected_idx]))
        spread = float(compute_spatial_spread(spatial_meta, selected_idx))

    score = normalized_objective(
        diversity=float(diversity),
        spread=float(spread),
        baseline_diversity=float(baseline_diversity),
        baseline_spread=float(baseline_spread),
        n_selected=int(selected_idx.size),
        target_n=int(target_n),
        weight_diversity=0.5,
        weight_spread=0.5,
        infeasible_penalty=0.1,
    )
    return {
        "objective_score": float(score.score),
        "objective_score_raw": float(score.raw_score),
        "objective_diversity_norm": float(score.diversity_norm),
        "objective_spread_norm": float(score.spread_norm),
        "objective_feasibility_ratio": float(score.feasibility_ratio),
        "objective_infeasible": bool(score.infeasible),
    }


def _try_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _nested_get(data: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_precomputed_features_fallback(*, repo_root: Path, n_rows: int) -> np.ndarray:
    candidates: list[Path] = []
    candidates.extend((repo_root / "outputs").glob("features-*.npy"))
    candidates.extend((repo_root / "outputs" / "runs").glob("features.npy"))
    candidates.extend(
        (repo_root / "outputs" / "runs").glob("**/parameter_resolution/features-*.npy")
    )

    matching: list[tuple[float, Path, np.ndarray]] = []
    for path in candidates:
        try:
            arr = np.load(path)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        if int(arr.shape[0]) == int(n_rows):
            matching.append((path.stat().st_mtime, path, arr))
    if not matching:
        raise FileNotFoundError(
            "No precomputed feature artifact found with row-count "
            f"{n_rows} for fallback loading."
        )
    matching.sort(key=lambda x: x[0], reverse=True)
    _mtime, chosen_path, chosen_arr = matching[0]
    print(
        "[WARN] Falling back to precomputed features artifact due missing images: "
        f"{chosen_path}"
    )
    return chosen_arr


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

    Direct in-process call to workflow-level run_optuna().
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

    metadata_path = ROOT / "data" / "new_all_tiles.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Required metadata CSV not found: {metadata_path}")

    # Run optimization directly
    run_optuna(
        n_trials=n_trials,
        n_candidates=n_candidates,
        n_samples=n_samples,
        n_samples_range=n_samples_range,
        metadata_path=metadata_path,
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
    from dataselector.data.metadata_source import canonical_metadata_path

    ROOT = _get_repo_root()
    csv_meta_path = canonical_metadata_path(ROOT)

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
                print(
                    f"[INFO] Auto-detected n_candidates from CSV: {n_candidates_local}"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "compare-samplers requires canonical metadata source at "
                    f"'{csv_meta_path}'."
                )
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
    seeds: Optional[Iterable[int]] = None,
    n_samples: Optional[int] = None,
    alpha_visual: Optional[float] = None,
    beta_spatial: Optional[float] = None,
    gamma_temporal: Optional[float] = None,
    min_distance_km: Optional[float] = None,
    report_label: Optional[str] = None,
) -> Path:
    ROOT = _get_repo_root()
    cfg_path = config_path or (ROOT / "config" / "pipeline_config.yaml")

    import yaml

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(
            f"[WARN] Config not found at {cfg_path}; using built-in defaults for comparison."
        )
        cfg = {}

    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection"), dict) else {}
    feat_cfg = (
        cfg.get("feature_extraction", {})
        if isinstance(cfg.get("feature_extraction"), dict)
        else {}
    )
    clu_cfg = (
        cfg.get("clustering", {}) if isinstance(cfg.get("clustering"), dict) else {}
    )

    resolved_n_samples = int(
        n_samples if n_samples is not None else sel_cfg.get("n_samples", 34)
    )
    resolved_alpha = float(
        alpha_visual
        if alpha_visual is not None
        else sel_cfg.get("alpha_visual", sel_cfg.get("weights", {}).get("alpha", 0.7))
    )
    resolved_beta = float(
        beta_spatial
        if beta_spatial is not None
        else sel_cfg.get("beta_spatial", sel_cfg.get("weights", {}).get("beta", 0.05))
    )
    resolved_gamma = float(
        gamma_temporal
        if gamma_temporal is not None
        else sel_cfg.get(
            "gamma_temporal", sel_cfg.get("weights", {}).get("gamma", 0.25)
        )
    )
    resolved_min_distance_km = float(
        min_distance_km
        if min_distance_km is not None
        else sel_cfg.get("min_distance_km", 40.0)
    )
    batch_size = int(feat_cfg.get("batch_size", 8))
    n_clusters_cfg = int(clu_cfg.get("n_clusters", 8))
    seed_panel = _resolve_seed_panel(seeds)

    if output_dir is not None:
        OUT = Path(output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        OUT = ROOT / "outputs" / "runs" / f"anchor_evidence_{ts}" / "isolated"
    OUT.mkdir(parents=True, exist_ok=True)

    # Runtime imports to avoid heavy deps at import-time
    from dataselector.analysis.metrics import compute_metrics
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.data.metadata_source import canonical_metadata_path
    from dataselector.data.spatial_schema import normalize_spatial_schema
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector
    from dataselector.workflows.objective_scoring import compute_baselines

    # Load cached features & metadata
    csv_meta = canonical_metadata_path(ROOT)
    metadata = load_metadata(str(csv_meta))
    feature_source = "load_or_extract_features"
    try:
        features = load_or_extract_features(
            OUT,
            csv_meta=str(csv_meta),
            batch_size=batch_size,
            cache=True,
            enforce_canonical=True,
        )
    except FileNotFoundError as exc:
        if "Missing image files" not in str(exc):
            raise
        features = _load_precomputed_features_fallback(
            repo_root=ROOT, n_rows=len(metadata)
        )
        feature_source = "precomputed_features_fallback"

    # compute cluster labels using existing pipeline to be consistent
    clustering = ClusteringPipeline(n_clusters=n_clusters_cfg)
    _, cluster_labels = clustering.fit_transform(features)
    baseline_diversity, baseline_spread = compute_baselines(
        features=features,
        metadata=metadata,
        metric="euclidean",
    )
    spatial_meta = normalize_spatial_schema(metadata, require_bounds=True, copy=True)

    rows: list[dict[str, Any]] = []

    # Two scenarios: baseline (no seed) and seeded (Hamburg)
    scenarios = [
        ("no_seed", None, None),
        ("seed_Hamburg_name", ["Hamburg"], None),
    ]

    for seed in seed_panel:
        for tag, pre_names, pre_idxs in scenarios:
            selector = DiversitySelector(
                n_samples=resolved_n_samples,
                use_multi_criteria=True,
                random_state=int(seed),
            )
            selected = selector.select(
                features=features,
                metadata=metadata,
                alpha_visual=resolved_alpha,
                beta_spatial=resolved_beta,
                gamma_temporal=resolved_gamma,
                spatial_constraint=True,
                min_distance_km=resolved_min_distance_km,
                pre_selected=pre_idxs,
                pre_selected_names=pre_names,
            )
            selected_idx = np.asarray(selected, dtype=int)

            metrics = compute_metrics(selected_idx, metadata, cluster_labels, features)
            objective_metrics = _compute_objective_for_selection(
                selector=selector,
                features=features,
                spatial_meta=spatial_meta,
                selected=selected_idx,
                baseline_diversity=baseline_diversity,
                baseline_spread=baseline_spread,
                target_n=resolved_n_samples,
            )
            metrics.update(objective_metrics)
            metrics.update(
                {
                    "scenario": tag,
                    "seed": int(seed),
                    "requested_n_samples": resolved_n_samples,
                    "alpha_visual": resolved_alpha,
                    "beta_spatial": resolved_beta,
                    "gamma_temporal": resolved_gamma,
                    "min_distance_km": resolved_min_distance_km,
                    "n_clusters": n_clusters_cfg,
                    "feature_source": feature_source,
                    "pre_selected_names": (
                        json.dumps(pre_names) if pre_names is not None else "[]"
                    ),
                    "pre_selected_indices": (
                        json.dumps(pre_idxs) if pre_idxs is not None else "[]"
                    ),
                    "selected_indices_json": json.dumps(
                        [int(i) for i in selected_idx.tolist()]
                    ),
                    "selection_signature": _selection_signature(selected_idx.tolist()),
                }
            )
            rows.append(metrics)

            # Persist per-seed selection snapshot for traceability.
            sel_df = metadata.iloc[selected_idx].copy()
            if len(sel_df) > 0:
                sel_df["cluster_label"] = cluster_labels[selected_idx]
            sel_df["selection_rank"] = range(len(sel_df))
            sel_df["scenario"] = tag
            sel_df["seed"] = int(seed)
            sel_df.to_csv(OUT / f"selection_{tag}_s{int(seed)}.csv", index=False)

    df = pd.DataFrame(rows).sort_values(["seed", "scenario"]).reset_index(drop=True)

    overlap_by_seed: dict[int, dict[str, float | int]] = {}
    pair_signatures: list[str] = []
    for seed in seed_panel:
        no_seed_row = df[(df["seed"] == int(seed)) & (df["scenario"] == "no_seed")]
        seeded_row = df[
            (df["seed"] == int(seed)) & (df["scenario"] == "seed_Hamburg_name")
        ]
        if len(no_seed_row) == 0 or len(seeded_row) == 0:
            overlap_by_seed[int(seed)] = {
                "overlap_count": np.nan,
                "swap_count": np.nan,
                "selection_jaccard": np.nan,
            }
            continue
        no_seed_selected = json.loads(str(no_seed_row.iloc[0]["selected_indices_json"]))
        seeded_selected = json.loads(str(seeded_row.iloc[0]["selected_indices_json"]))
        overlap_by_seed[int(seed)] = _selection_overlap_metrics(
            no_seed_selected,
            seeded_selected,
        )
        pair_signatures.append(
            f"{no_seed_row.iloc[0]['selection_signature']}||{seeded_row.iloc[0]['selection_signature']}"
        )

    for metric_name in ("overlap_count", "swap_count", "selection_jaccard"):
        df[metric_name] = df["seed"].map(
            lambda s: overlap_by_seed.get(int(s), {}).get(metric_name, np.nan)
        )

    effective_by_scenario = (
        df.groupby("scenario")["selection_signature"].nunique().to_dict()
    )
    effective_unseeded = int(effective_by_scenario.get("no_seed", 0))
    effective_seeded = int(effective_by_scenario.get("seed_Hamburg_name", 0))
    effective_pairs = int(len(set(pair_signatures)))
    inference_status = (
        "independent"
        if min(effective_unseeded, effective_seeded, effective_pairs) >= 2
        else "non_independent_seed_replay"
    )

    df["effective_replicates_scenario"] = df["scenario"].map(
        lambda s: int(effective_by_scenario.get(str(s), 0))
    )
    df["effective_replicates_no_seed"] = effective_unseeded
    df["effective_replicates_seed_Hamburg_name"] = effective_seeded
    df["effective_replicate_pairs"] = effective_pairs
    df["nominal_replicates"] = int(len(seed_panel))
    df["inference_status"] = inference_status

    summary_df = _summarize_endpoints_by_scenario(df, endpoints=ALL_ENDPOINTS)
    stats_df = _compute_paired_endpoint_stats(
        df,
        endpoints=ALL_ENDPOINTS,
        inference_status=inference_status,
    )

    raw_csv = OUT / "seed_vs_unseed_metrics.csv"
    summary_csv = OUT / "seed_vs_unseed_summary.csv"
    stats_csv = OUT / "seed_vs_unseed_stats.csv"
    report_md = OUT / "seed_vs_unseed_report.md"
    df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    overlap_df = (
        df[["seed", "overlap_count", "swap_count", "selection_jaccard"]]
        .drop_duplicates(subset=["seed"])
        .sort_values("seed")
        .reset_index(drop=True)
    )

    heading = report_label or "Hamburg Anchor: Seeded vs Unseeded (Isolated)"
    with report_md.open("w", encoding="utf-8") as f:
        f.write(f"# {heading}\n\n")
        f.write("- Comparison design: paired by seed (`delta = seeded - unseeded`)\n")
        f.write(f"- Seeds: `{seed_panel}`\n")
        f.write(
            f"- Fixed params: n_samples={resolved_n_samples}, alpha={resolved_alpha}, "
            f"beta={resolved_beta}, gamma={resolved_gamma}, "
            f"min_distance_km={resolved_min_distance_km}, n_clusters={n_clusters_cfg}\n"
        )
        f.write(f"- Primary endpoint: `{PRIMARY_ENDPOINT}`\n\n")
        f.write(f"- Feature source: `{feature_source}`\n\n")
        f.write("## Independence Check\n\n")
        f.write(f"- inference_status: `{inference_status}`\n")
        f.write(f"- nominal_replicates: `{len(seed_panel)}`\n")
        f.write(f"- effective_replicates (no_seed): `{effective_unseeded}`\n")
        f.write(f"- effective_replicates (seed_Hamburg_name): `{effective_seeded}`\n")
        f.write(f"- effective_replicate_pairs: `{effective_pairs}`\n")
        if inference_status != "independent":
            f.write(
                "- Inferential statistics are marked as non-inferential for this seed replay panel.\n"
            )
        f.write("\n## Selection Overlap by Seed\n\n")
        try:
            f.write(overlap_df.to_markdown(index=False))
        except Exception:
            f.write("```text\n")
            f.write(overlap_df.to_string(index=False))
            f.write("\n```\n")
        f.write("## Paired Statistics\n\n")
        if len(stats_df) > 0:
            try:
                f.write(stats_df.to_markdown(index=False))
            except Exception:
                f.write("```text\n")
                f.write(stats_df.to_string(index=False))
                f.write("\n```\n")
        else:
            f.write("No paired endpoint statistics available.\n")
        f.write("\n\n## Scenario Summary\n\n")
        if len(summary_df) > 0:
            try:
                f.write(summary_df.to_markdown(index=False))
            except Exception:
                f.write("```text\n")
                f.write(summary_df.to_string(index=False))
                f.write("\n```\n")
        else:
            f.write("No summary rows available.\n")
        f.write("\n\n## Artifacts\n\n")
        f.write(f"- Raw metrics: `{raw_csv.name}`\n")
        f.write(f"- Summary: `{summary_csv.name}`\n")
        f.write(f"- Stats: `{stats_csv.name}`\n")
        f.write(
            "- Selection snapshots: `selection_<scenario>_s<seed>.csv` for all seeds\n"
        )

    print("Done. Results:", OUT)
    return OUT


def compare_production_runs_quick_delta(
    *,
    seeded_run_dir: Path,
    unseeded_run_dir: Path,
    output_dir: Path,
) -> Path:
    seeded_dir = Path(seeded_run_dir)
    unseeded_dir = Path(unseeded_run_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeded_run_meta = _try_read_json(seeded_dir / "run_metadata.json")
    unseeded_run_meta = _try_read_json(unseeded_dir / "run_metadata.json")
    seeded_tuning_meta = _try_read_json(seeded_dir / "tuning_weights" / "meta.json")
    unseeded_tuning_meta = _try_read_json(unseeded_dir / "tuning_weights" / "meta.json")

    seeded_best = seeded_tuning_meta.get("best_metrics", {})
    unseeded_best = unseeded_tuning_meta.get("best_metrics", {})
    if not isinstance(seeded_best, dict):
        seeded_best = {}
    if not isinstance(unseeded_best, dict):
        unseeded_best = {}

    def _val_num(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    seeded_n_samples = _nested_get(seeded_run_meta, ["extra", "n_samples"], None)
    unseeded_n_samples = _nested_get(unseeded_run_meta, ["extra", "n_samples"], None)
    seeded_min_dist_arr = _nested_get(
        seeded_run_meta, ["extra", "validation_min_distances"], []
    )
    unseeded_min_dist_arr = _nested_get(
        unseeded_run_meta, ["extra", "validation_min_distances"], []
    )
    seeded_min_dist = (
        float(seeded_min_dist_arr[0])
        if isinstance(seeded_min_dist_arr, list) and seeded_min_dist_arr
        else float("nan")
    )
    unseeded_min_dist = (
        float(unseeded_min_dist_arr[0])
        if isinstance(unseeded_min_dist_arr, list) and unseeded_min_dist_arr
        else float("nan")
    )

    endpoint_rows: list[dict[str, Any]] = []
    metric_names = [
        "n_selected",
        "clusters_covered",
        "spatial_mean_km",
        "spatial_min_km",
        "temporal_std",
        "wwi_percent",
        "alpha",
        "beta",
        "gamma",
    ]
    for metric in metric_names:
        seeded_val = seeded_best.get(metric, np.nan)
        unseeded_val = unseeded_best.get(metric, np.nan)
        seeded_num = _val_num(seeded_val)
        unseeded_num = _val_num(unseeded_val)
        delta = (
            float(seeded_num - unseeded_num)
            if np.isfinite(seeded_num) and np.isfinite(unseeded_num)
            else float("nan")
        )
        endpoint_rows.append(
            {
                "metric": metric,
                "seeded_value": seeded_val,
                "unseeded_value": unseeded_val,
                "delta_seeded_minus_unseeded": delta,
                "is_policy_context": False,
            }
        )

    endpoint_rows.extend(
        [
            {
                "metric": "n_samples",
                "seeded_value": seeded_n_samples,
                "unseeded_value": unseeded_n_samples,
                "delta_seeded_minus_unseeded": (
                    _val_num(seeded_n_samples) - _val_num(unseeded_n_samples)
                    if np.isfinite(_val_num(seeded_n_samples))
                    and np.isfinite(_val_num(unseeded_n_samples))
                    else float("nan")
                ),
                "is_policy_context": True,
            },
            {
                "metric": "validation_min_distance_km",
                "seeded_value": seeded_min_dist,
                "unseeded_value": unseeded_min_dist,
                "delta_seeded_minus_unseeded": (
                    float(seeded_min_dist - unseeded_min_dist)
                    if np.isfinite(seeded_min_dist) and np.isfinite(unseeded_min_dist)
                    else float("nan")
                ),
                "is_policy_context": True,
            },
            {
                "metric": "pre_selected_names",
                "seeded_value": json.dumps(
                    seeded_tuning_meta.get(
                        "pre_selected_names",
                        _nested_get(
                            seeded_run_meta, ["extra", "pre_selected_names"], []
                        ),
                    )
                ),
                "unseeded_value": json.dumps(
                    unseeded_tuning_meta.get(
                        "pre_selected_names",
                        _nested_get(
                            unseeded_run_meta, ["extra", "pre_selected_names"], []
                        ),
                    )
                ),
                "delta_seeded_minus_unseeded": float("nan"),
                "is_policy_context": True,
            },
        ]
    )

    delta_df = pd.DataFrame(endpoint_rows)
    delta_csv = out_dir / "production_delta.csv"
    delta_md = out_dir / "production_delta_report.md"
    delta_df.to_csv(delta_csv, index=False)

    with delta_md.open("w", encoding="utf-8") as f:
        f.write("# Production Quick Delta (Existing Full Runs)\n\n")
        f.write(
            "- Purpose: contextual evidence from existing full runs, not a causal isolated test\n"
        )
        f.write(f"- Seeded run: `{seeded_dir}`\n")
        f.write(f"- Unseeded run: `{unseeded_dir}`\n")
        f.write(
            f"- Policy context: n_samples seeded={seeded_n_samples}, unseeded={unseeded_n_samples}\n"
        )
        f.write(
            f"- Policy context: validation_min_distance_km seeded={seeded_min_dist}, "
            f"unseeded={unseeded_min_dist}\n\n"
        )
        f.write(
            "Interpretation note: differences can include policy/config deltas and should "
            "not be over-interpreted as isolated Hamburg-anchor effects.\n\n"
        )
        f.write("## Delta Table\n\n")
        try:
            f.write(delta_df.to_markdown(index=False))
        except Exception:
            f.write("```text\n")
            f.write(delta_df.to_string(index=False))
            f.write("\n```\n")

    return out_dir


def benchmark_seed(
    seeds: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
    subset_n: int = 200,
) -> Path:
    from dataselector.data.metadata_source import canonical_metadata_path

    ROOT = _get_repo_root()
    OUT = output_dir or (ROOT / "outputs")
    OUT.mkdir(exist_ok=True, parents=True)

    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.clustering import ClusteringPipeline

    # Ensure features/metadata exists or extract on-the-fly
    csv_meta = canonical_metadata_path(ROOT)
    features = load_or_extract_features(
        out_dir=OUT,
        csv_meta=str(csv_meta),
        batch_size=16,
        cache=True,
        enforce_canonical=True,
    )
    _metadata = load_metadata(str(csv_meta))

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
            "type": bool,
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
    raise SystemExit(main())
