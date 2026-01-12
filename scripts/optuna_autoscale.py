# ruff: noqa: E402
"""Staged Optuna runner that progressively refines search and stops on convergence.

Usage:
    python scripts/optuna_autoscale.py --n-candidates 500 --stages 50 100 300 full --trials 20 40 80 160

Behavior:
- Runs Optuna in stages for increasing `n_samples` (e.g., 50 -> 100 -> 300 -> full).
- After each stage, narrows the search ranges around the best parameters found.
- Stops early if best parameters converge (parameter deltas < tolerance) for `patience` stages.
- Saves per-stage summaries and a dated report in `outputs/`.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diversity_selector import DiversitySelector

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def load_or_create_data(n=500, dim=256, seed=123):
    features_path = OUT / "features.npy"
    metadata_path = OUT / "metadata.csv"

    from src.io import load_or_extract_features

    if features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=OUT, csv_meta=str(metadata_path), batch_size=16, cache=False
        )
        metadata = pd.read_csv(metadata_path)
    else:
        rng = np.random.RandomState(seed)
        features = rng.randn(n, dim).astype("float32")
        metadata = pd.DataFrame(
            {
                "N": np.random.uniform(48, 55, n),
                "left": np.random.uniform(6, 15, n),
                "year": np.random.randint(1880, 1945, n),
            }
        )

    return features, metadata


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_objective(features, metadata, n_samples, min_distance_bounds):
    def objective(trial: optuna.trial.Trial):
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

        selector = DiversitySelector(n_samples=n_samples, use_multi_criteria=True)
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_dist,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
        )

        n_selected = len(selected)
        if n_selected == 0:
            return 0.0

        diversity = selector._calculate_diversity_score(features[selected])
        spatial_spread = metadata.loc[selected, ["N", "left"]].std().mean()
        score = diversity * spatial_spread

        trial.set_user_attr("alpha", float(alpha))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("gamma", float(gamma))
        trial.set_user_attr("min_distance_km", int(min_dist))
        trial.set_user_attr("n_selected", int(n_selected))
        trial.set_user_attr("diversity", float(diversity))
        trial.set_user_attr("spatial_spread", float(spatial_spread))

        return float(score)

    return objective


def run_autoscale(
    n_trials_per_stage, stages_samples, features, metadata, patience=2, tol=None
):
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
        objective = make_objective(features, metadata, n_samples, bounds)

        # Record starting trial count to slice per-stage trials later
        start_trial_count = len(study.trials)
        study.optimize(objective, n_trials=n_trials_per_stage[stage_idx])
        end_trial_count = len(study.trials)

        # Extract trials for this stage
        stage_trials = study.trials[start_trial_count:end_trial_count]

        # Plot per-stage diagnostics
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            date = datetime.now().strftime("%Y%m%d")
            # Objective history for this stage
            values = [t.value for t in stage_trials if t.value is not None]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(len(values)), values, marker="o")
                ax.set_xlabel("trial (stage)")
                ax.set_ylabel("value")
                ax.set_title(f"Stage {stage_idx+1} objective")
                out = OUT / f"optuna_autoscale_stage{stage_idx+1}_history_{date}.png"
                fig.savefig(out, bbox_inches="tight")
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
                out = (
                    OUT / f"optuna_autoscale_stage{stage_idx+1}_min_distance_{date}.png"
                )
                fig.savefig(out, bbox_inches="tight")
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
                out = OUT / f"optuna_autoscale_stage{stage_idx+1}_alpha_beta_{date}.png"
                fig.savefig(out, bbox_inches="tight")
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

        # Save stage summary
        hist = {
            "stage": stage_idx,
            "n_samples": n_samples,
            "trials": n_trials_per_stage[stage_idx],
            **best_params,
        }
        history.append(hist)

        # Check convergence
        if best_prev is not None:
            # Compute absolute deltas
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

        # Narrow bounds around best for next stage (progressive refinement)
        # For raw a,b,c use +/- 0.2 of normalized alpha (convert back to raw scale roughly)
        alpha = best_params["alpha"]
        beta = best_params["beta"]
        gamma = best_params["gamma"]

        # convert normalized to raw by assuming total ~1 and split into similar magnitudes
        # We'll create tight bounds around alpha/beta/gamma in raw space using small +/- window
        for key, val in [("a", alpha), ("b", beta), ("c", gamma)]:
            lo = clamp(val - 0.2, 0.01, 1.0)
            hi = clamp(val + 0.2, 0.01, 1.0)
            bounds[key] = (lo, hi)

        # Narrow min_distance bounds
        md = best_params["min_distance_km"]
        if md is None:
            md = 28
        bounds["min_distance_km"] = (max(0, int(md - 10)), int(md + 10))

        print("New bounds for next stage:", bounds)

    # Persist history
    date = datetime.now().strftime("%Y%m%d")
    summary_df = pd.DataFrame(history)
    summary_file = OUT / f"optuna_autoscale_summary_{date}.csv"
    summary_df.to_csv(summary_file, index=False)

    # Create a short markdown report including per-stage summary and links to plots
    report_md = OUT / f"optuna_autoscale_report_{date}.md"
    lines = [f"# Optuna Auto-scale Report ({date})", "", "## Stages summary", ""]
    try:
        # to_markdown may require 'tabulate' package; fall back to CSV if missing
        lines.append(summary_df.to_markdown(index=False))
    except Exception:
        csv_summary = OUT / f"optuna_autoscale_summary_{date}.csv"
        lines.append(
            f"(could not render markdown table; saved CSV summary: {csv_summary.name})"
        )
        summary_df.to_csv(csv_summary, index=False)

    lines.append("")

    # List generated stage plots
    lines.append("## Stage plots")
    for f in sorted(OUT.glob(f"optuna_autoscale_stage*_{date}.png")):
        lines.append(f"- {f.name}")

    report_md.write_text("\n".join(lines))
    print("Report written to", report_md)

    # Save best trial and study
    best_overall = study.best_trial
    out_best = OUT / f"optuna_autoscale_best_{date}.json"
    with out_best.open("w") as fh:
        json.dump(
            {
                "value": best_overall.value,
                "params": best_overall.params,
                "user_attrs": best_overall.user_attrs,
            },
            fh,
            indent=2,
        )

    # Save study as pickle if joblib available
    try:
        import joblib

        joblib.dump(study, OUT / f"optuna_autoscale_study_{date}.pkl")
    except Exception:
        print("joblib not available; not saving study object")

    print("Auto-scale optimization finished. Summary:", summary_file)
    return summary_file, out_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, nargs="+", default=[20, 40, 80, 160])
    parser.add_argument("--stages", nargs="+", default=["50", "100", "300", "full"])
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)

    args = parser.parse_args()

    features, metadata = load_or_create_data(
        n=args.n_candidates, dim=args.dim, seed=args.seed
    )

    # interpret stages
    stages = []
    for s in args.stages:
        if s == "full":
            stages.append(min(args.n_candidates, 1000))
        else:
            stages.append(int(s))

    # ensure trials len matches stages len
    if len(args.n_trials) != len(stages):
        # if single value provided, repeat
        if len(args.n_trials) == 1:
            n_trials_per_stage = [args.n_trials[0]] * len(stages)
        else:
            raise SystemExit("Provide n-trials per stage or a single value.")
    else:
        n_trials_per_stage = args.n_trials

    run_autoscale(
        n_trials_per_stage, stages, features, metadata, patience=args.patience
    )
