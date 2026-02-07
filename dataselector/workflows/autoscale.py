# ruff: noqa: E402
"""Staged Optuna runner that progressively refines search and stops on convergence.

Usage:
    dataselector autoscale --csv data/new_all_tiles.csv --stages 50 100 300 full --output-dir outputs/ --n-trials 20 40 80 160 --n-candidates 500 --seed 42

Behavior:
- Runs Optuna in stages for increasing `n_samples` (e.g., 50 -> 100 -> 300 -> full).
- After each stage, narrows the search ranges around the best parameters found.
- Stops early if best parameters converge (parameter deltas < tolerance) for `patience` stages.
- Saves per-stage summaries and a dated report in `<output-dir>/`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command
from dataselector.data.spatial_schema import (
    normalize_spatial_schema,
)
from dataselector.data.spatial_schema import spatial_spread as compute_spatial_spread


def clamp(v, lo, hi):
    """Clamp value v to range [lo, hi]."""
    return max(lo, min(hi, v))


def load_or_create_data(csv_meta=None, n=None, dim=256, seed=123, out_dir=None):
    """Load features and metadata from cache or extract/generate them.

    Args:
        csv_meta: Path to CSV metadata file
        n: Number of samples (fallback if not loading from CSV)
        dim: Feature dimension (fallback)
        seed: Random seed
        out_dir: Output directory for cache files

    Returns:
        features (np.ndarray), metadata (pd.DataFrame)
    """
    out_dir = Path(out_dir or "outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    features_path = out_dir / "features.npy"
    metadata_path = out_dir / "metadata.csv" if csv_meta is None else Path(csv_meta)

    from dataselector.data.io import load_metadata, load_or_extract_features

    if features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=out_dir, csv_meta=str(metadata_path), batch_size=16, cache=True
        )
        metadata = load_metadata(str(metadata_path))
    else:
        # Fallback: generate synthetic data
        rng = np.random.RandomState(seed)
        if n is None:
            n = 673
        features = rng.randn(n, dim).astype("float32")
        center_x = rng.uniform(450000, 650000, n)
        center_y = rng.uniform(5800000, 6100000, n)
        half_w = rng.uniform(40, 80, n)
        half_h = rng.uniform(40, 80, n)
        metadata = pd.DataFrame(
            {
                "ul_x": center_x - half_w,
                "ul_y": center_y + half_h,
                "lr_x": center_x + half_w,
                "lr_y": center_y - half_h,
                "year": rng.randint(1880, 1945, n),
            }
        )

    return features, metadata


def make_objective(
    features,
    metadata,
    n_samples,
    min_distance_bounds,
    pre_selected_names=None,
    pre_selected_indices=None,
):
    """Create objective function for Optuna trial.

    Args:
        features: Feature matrix
        metadata: Metadata DataFrame
        n_samples: Number of samples to select
        min_distance_bounds: Dict with keys 'a', 'b', 'c', 'min_distance_km'
        pre_selected_names: Optional list of pre-selected tile names
        pre_selected_indices: Optional list of pre-selected tile indices

    Returns:
        objective function callable
    """

    def objective(trial):
        a = trial.suggest_float("a", *min_distance_bounds["a"])
        b = trial.suggest_float("b", *min_distance_bounds["b"])
        c = trial.suggest_float("c", *min_distance_bounds["c"])
        total = a + b + c
        alpha = a / total
        beta = b / total
        gamma = c / total

        min_dist = trial.suggest_int(
            "min_distance_km", *min_distance_bounds["min_distance_km"]
        )

        from dataselector.selection.diversity_selector import DiversitySelector

        selector = DiversitySelector(n_samples=n_samples, use_multi_criteria=True)
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_dist,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
            pre_selected=pre_selected_indices,
            pre_selected_names=pre_selected_names,
        )

        n_selected = len(selected)
        if n_selected == 0:
            return 0.0

        diversity = selector._calculate_diversity_score(features[selected])
        spatial_meta = normalize_spatial_schema(
            metadata, require_bounds=True, copy=True
        )
        spread = compute_spatial_spread(spatial_meta, selected)
        score = diversity * spread

        trial.set_user_attr("alpha", float(alpha))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("gamma", float(gamma))
        trial.set_user_attr("min_distance_km", int(min_dist))
        trial.set_user_attr("n_selected", int(n_selected))
        trial.set_user_attr("diversity", float(diversity))
        trial.set_user_attr("spatial_spread", float(spread))
        trial.set_user_attr("n_samples", int(n_samples))

        return float(score)

    return objective


def run_autoscale(
    n_trials_per_stage,
    stages_samples,
    features,
    metadata,
    out_dir,
    patience=2,
    tol=None,
    pre_names=None,
    pre_indices=None,
):
    """Run multi-stage Optuna optimization with progressive refinement.

    Args:
        n_trials_per_stage: List of trial counts per stage
        stages_samples: List of n_samples values per stage
        features: Feature matrix
        metadata: Metadata DataFrame
        out_dir: Output directory for results
        patience: Number of stages without improvement to trigger early stopping
        tol: Tolerance dict for convergence detection
        pre_names: Optional pre-selected tile names
        pre_indices: Optional pre-selected tile indices

    Returns:
        summary_file (Path), best_file (Path)
    """
    try:
        import optuna
    except ImportError:
        print(
            "Error: optuna is required to run autoscale. Install optuna in your environment."
        )
        raise SystemExit(2)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Initialize global search bounds
    bounds = {
        "a": (0.01, 1.0),
        "b": (0.01, 1.0),
        "c": (0.01, 1.0),
        "min_distance_km": (0, 60),
    }

    study = optuna.create_study(direction="maximize")

    history = []
    best_prev = None
    no_change = 0

    for stage_idx, n_samples in enumerate(stages_samples):
        print(
            f"\n=== Stage {stage_idx+1}/{len(stages_samples)}: n_samples={n_samples} | trials={n_trials_per_stage[stage_idx]} ==="
        )
        objective = make_objective(
            features,
            metadata,
            n_samples,
            bounds,
            pre_selected_names=pre_names,
            pre_selected_indices=pre_indices,
        )

        start_trial_count = len(study.trials)
        study.optimize(objective, n_trials=n_trials_per_stage[stage_idx])
        end_trial_count = len(study.trials)

        stage_trials = study.trials[start_trial_count:end_trial_count]

        # Plot per-stage diagnostics
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            date = datetime.now().strftime("%Y%m%d")

            # Objective history
            values = [t.value for t in stage_trials if t.value is not None]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(len(values)), values, marker="o")
                ax.set_xlabel("trial (stage)")
                ax.set_ylabel("value")
                ax.set_title(f"Stage {stage_idx+1} objective")
                out_file = out_dir / f"autoscale_stage{stage_idx+1}_history_{date}.png"
                fig.savefig(out_file, bbox_inches="tight")
                plt.close(fig)

            # min_distance distribution
            md_vals = [
                t.user_attrs.get("min_distance_km")
                for t in stage_trials
                if "min_distance_km" in t.user_attrs
            ]
            if md_vals:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(md_vals, bins=10, kde=False, ax=ax)
                ax.set_title(f"Stage {stage_idx+1} min_distance distribution")
                out_file = (
                    out_dir / f"autoscale_stage{stage_idx+1}_min_distance_{date}.png"
                )
                fig.savefig(out_file, bbox_inches="tight")
                plt.close(fig)

            # parameter scatter (alpha vs beta)
            alf = [
                t.user_attrs.get("alpha")
                for t in stage_trials
                if "alpha" in t.user_attrs
            ]
            bet = [
                t.user_attrs.get("beta") for t in stage_trials if "beta" in t.user_attrs
            ]
            if alf and bet:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=alf, y=bet, ax=ax)
                ax.set_xlabel("alpha")
                ax.set_ylabel("beta")
                ax.set_title(f"Stage {stage_idx+1} alpha vs beta")
                out_file = (
                    out_dir / f"autoscale_stage{stage_idx+1}_alpha_beta_{date}.png"
                )
                fig.savefig(out_file, bbox_inches="tight")
                plt.close(fig)

            print(f"Wrote stage plots for stage {stage_idx+1}")
        except Exception as e:
            print("Warning: could not create stage plots:", e)

        best = study.best_trial
        best_params = {
            "value": best.value,
            "alpha": best.user_attrs.get("alpha"),
            "beta": best.user_attrs.get("beta"),
            "gamma": best.user_attrs.get("gamma"),
            "min_distance_km": best.user_attrs.get("min_distance_km"),
        }

        print("Stage best:", best_params)

        hist = {
            "stage": stage_idx,
            "n_samples": n_samples,
            "trials": n_trials_per_stage[stage_idx],
            **best_params,
        }
        history.append(hist)

        # Check convergence
        if best_prev is not None:
            dalpha = abs(best_prev["alpha"] - best_params["alpha"])
            dbeta = abs(best_prev["beta"] - best_params["beta"])
            dmin = abs(best_prev["min_distance_km"] - best_params["min_distance_km"])
            dval = abs(best_prev["value"] - best_params["value"])

            print(
                f"deltas: dalpha={dalpha:.4f}, dbeta={dbeta:.4f}, dmin={dmin:.2f}, dval={dval:.4f}"
            )

            tol = tol or {
                "alpha": 0.02,
                "beta": 0.02,
                "min_distance_km": 1.0,
                "value": 1e-3,
            }

            if (
                dalpha < tol["alpha"]
                and dbeta < tol["beta"]
                and dmin < tol["min_distance_km"]
            ):
                no_change += 1
                print(f"No significant change detected ({no_change}/{patience})")
            else:
                no_change = 0

            if no_change >= patience:
                print("Converged: stopping auto-scale optimization")
                break

        best_prev = best_params

        # Narrow bounds around best for next stage
        alpha = best_params["alpha"]
        beta = best_params["beta"]
        gamma = best_params["gamma"]

        for key, val in [("a", alpha), ("b", beta), ("c", gamma)]:
            lo = clamp(val - 0.2, 0.01, 1.0)
            hi = clamp(val + 0.2, 0.01, 1.0)
            bounds[key] = (lo, hi)

        md = best_params["min_distance_km"]
        if md is None:
            md = 28
        bounds["min_distance_km"] = (max(0, int(md - 10)), int(md + 10))

        print("New bounds for next stage:", bounds)

    # Persist history
    date = datetime.now().strftime("%Y%m%d")
    summary_df = pd.DataFrame(history)
    summary_file = out_dir / f"autoscale_summary_{date}.csv"
    summary_df.to_csv(summary_file, index=False)

    # Create markdown report
    report_md = out_dir / f"autoscale_report_{date}.md"
    lines = [f"# Autoscale Report ({date})", "", "## Stages summary", ""]
    try:
        lines.append(summary_df.to_markdown(index=False))
    except Exception:
        csv_summary = out_dir / f"autoscale_summary_{date}.csv"
        lines.append(
            f"(could not render markdown table; saved CSV summary: {csv_summary.name})"
        )
        summary_df.to_csv(csv_summary, index=False)

    lines.append("")
    lines.append("## Stage plots")
    for f in sorted(out_dir.glob(f"autoscale_stage*_{date}.png")):
        lines.append(f"- {f.name}")

    report_md.write_text("\n".join(lines))
    print("Report written to", report_md)

    # Save best trial
    best_overall = study.best_trial
    out_best = out_dir / f"autoscale_best_{date}.json"
    with out_best.open("w") as fh:
        json.dump(
            {
                "best_value": best_overall.value,
                "best_params": best_overall.params,
                "user_attrs": best_overall.user_attrs,
                "n_trials": len(study.trials),
            },
            fh,
            indent=2,
        )

    # Write canonical 'latest' best file
    latest = out_dir / "autoscale_best_latest.json"
    with latest.open("w") as fh:
        json.dump(
            {
                "best_value": best_overall.value,
                "best_params": best_overall.params,
                "user_attrs": best_overall.user_attrs,
                "n_trials": len(study.trials),
            },
            fh,
            indent=2,
        )

    # Extract n_samples and write simple text file
    best_n_samples = best_overall.user_attrs.get("n_samples")
    sel_file = out_dir / "autoscale_selected_n_samples.txt"
    if best_n_samples is not None:
        sel_file.write_text(str(int(best_n_samples)))
        print(f"BEST_N_SAMPLES: {int(best_n_samples)}")
    else:
        try:
            hist_df = pd.read_csv(summary_file)
            idx = hist_df["value"].idxmax()
            inferred = int(hist_df.loc[idx, "n_samples"])
            sel_file.write_text(str(inferred))
            print(f"BEST_N_SAMPLES (inferred): {inferred}")
        except Exception:
            print("WARNING: could not determine best n_samples")

    # Save study as pickle if joblib available
    try:
        import joblib

        joblib.dump(study, out_dir / f"autoscale_study_{date}.pkl")
    except Exception:
        print("joblib not available; not saving study object")

    print("Auto-scale optimization finished. Summary:", summary_file)
    return summary_file, out_best


@cli_command(
    "autoscale",
    help="Staged Optuna runner with progressive refinement and convergence detection",
    args={
        "csv": {
            "type": str,
            "default": None,
            "help": "Path to CSV metadata file (default: generate synthetic)",
        },
        "n_trials": {
            "type": int,
            "nargs": "+",
            "default": [20, 40, 80, 160],
            "help": "Number of trials per stage",
        },
        "stages": {
            "type": str,
            "nargs": "+",
            "default": ["50", "100", "300", "full"],
            "help": "Sample sizes per stage ('full' = all candidates)",
        },
        "output_dir": {
            "type": str,
            "default": "outputs",
            "help": "Output directory for results",
        },
        "n_candidates": {
            "type": int,
            "default": None,
            "help": "Number of candidates (fallback if CSV not provided)",
        },
        "dim": {
            "type": int,
            "default": 256,
            "help": "Feature dimension",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Random seed",
        },
        "patience": {
            "type": int,
            "default": 2,
            "help": "Number of stages without improvement to trigger early stopping",
        },
        "pre_names": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Optional pre-selected tile names",
        },
        "pre_indices": {
            "type": int,
            "nargs": "*",
            "default": None,
            "help": "Optional pre-selected tile indices",
        },
    },
)
def main(
    csv: str | None = None,
    n_trials: list[int] | None = None,
    stages: list[str] | None = None,
    output_dir: str = "outputs",
    n_candidates: int | None = None,
    dim: int = 256,
    seed: int = 42,
    patience: int = 2,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
) -> int:
    """Entry point for autoscale command.

    Runs Optuna in stages for increasing n_samples, progressively refining
    search ranges around best parameters found. Stops early if convergence
    detected (parameter deltas < tolerance) for `patience` stages.
    """
    # Default values
    if n_trials is None:
        n_trials = [20, 40, 80, 160]
    if stages is None:
        stages = ["50", "100", "300", "full"]

    features, metadata = load_or_create_data(
        csv_meta=csv,
        n=n_candidates,
        dim=dim,
        seed=seed,
        out_dir=output_dir,
    )

    actual_n = len(features)

    # Interpret stages
    interpreted_stages = []
    for s in stages:
        if s == "full":
            interpreted_stages.append(actual_n)
        else:
            interpreted_stages.append(int(s))

    # Ensure trials len matches stages len
    if len(n_trials) != len(interpreted_stages):
        if len(n_trials) == 1:
            n_trials_per_stage = [n_trials[0]] * len(interpreted_stages)
        else:
            print("Error: Provide n-trials per stage or a single value.")
            return 1
    else:
        n_trials_per_stage = n_trials

    run_autoscale(
        n_trials_per_stage,
        interpreted_stages,
        features,
        metadata,
        out_dir=output_dir,
        patience=patience,
        pre_names=pre_names,
        pre_indices=pre_indices,
    )

    return 0
