#!/usr/bin/env python3
"""Apply best Bootstrap candidate into pipeline config.

Usage:
  ./scripts/exec_in_env.sh --env dataselector -- python scripts/apply_bootstrap_best.py --bootstrap-summary outputs/fine_sweep/bootstrap_summary.csv --inject
  ./scripts/exec_in_env.sh --env dataselector -- python scripts/apply_bootstrap_best.py --bootstrap-summary outputs/bootstrap_results_summary.csv --write-config config/pipeline_config.bootstrap.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

# STARTUP ENV VALIDATION
try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dataselector.compat import validate_environment_full
    # Allow pytest/tests to bypass env check via env var
    if "--skip-env-check" not in sys.argv and os.environ.get("DATASELECTOR_IGNORE_ENV_CHECK") != "1":
        validate_environment_full()
except Exception as e:
    # Only exit if not in test mode
    if os.environ.get("DATASELECTOR_IGNORE_ENV_CHECK") != "1":
        print(f"\n❌ STARTUP VALIDATION FAILED:\n{e}\n", file=sys.stderr)
        print("Fix: ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages 'numpy==1.26.4 numba==0.63.1' --yes -- python scripts/apply_bootstrap_best.py", file=sys.stderr)
        sys.exit(1)


def _normalize_and_aggregate_bootstrap_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and, if necessary, aggregate per-iteration data into summary rows.

    The function accepts either:
      - a per-candidate summary dataframe with columns ending in _mean/_std
      - or a per-bootstrap-iteration dataframe with columns like 'alpha','beta','gamma','temporal_std','wwi_percent','jaccard_with_original'

    It returns a dataframe with the following guaranteed columns:
      ['alpha','beta','gamma','min_distance_km',
       'temporal_std_mean','temporal_std_std',
       'wwi_percent_mean','wwi_percent_std',
       'jaccard_with_original_mean','jaccard_with_original_std']
    """
    df = df.copy()

    # Accept aliases for parameter columns
    for a in ["alpha", "a", "params_a", "attrs_a"]:
        if a in df.columns:
            df["alpha"] = df[a]
            break
    for b in ["beta", "b", "params_b", "attrs_b"]:
        if b in df.columns:
            df["beta"] = df[b]
            break
    for c in ["gamma", "c", "params_c", "attrs_c"]:
        if c in df.columns:
            df["gamma"] = df[c]
            break
    for d in ["min_distance_km", "min_distance", "params_min_distance_km"]:
        if d in df.columns:
            df["min_distance_km"] = df[d]
            break

    # Handle describe()-style summaries (rows: count, mean, std, ...) -> convert to single-row aggregated format
    if "Unnamed: 0" in df.columns and set(["count", "mean", "std"]).issubset(set(df["Unnamed: 0"].astype(str))):
        # extract numeric metric columns
        value_cols = [c for c in df.columns if c != "Unnamed: 0"]
        mean_row = df[df["Unnamed: 0"] == "mean"].iloc[0]
        std_row = df[df["Unnamed: 0"] == "std"].iloc[0]
        out = {}
        for c in value_cols:
            out[f"{c}_mean"] = float(mean_row[c]) if pd.notna(mean_row[c]) else float("nan")
            out[f"{c}_std"] = float(std_row[c]) if pd.notna(std_row[c]) else float("nan")
        # convert to DataFrame with one row
        return pd.DataFrame([out])

    # If already aggregated (has *_mean), ensure required *_mean/_std columns exist
    mean_cols = [col for col in df.columns if col.endswith("_mean")]
    if mean_cols:
        # Map common names to expected canonical names
        # temporal_std_mean <- temporal_std_mean or temporal_std
        if "temporal_std_mean" not in df.columns:
            if "temporal_std" in df.columns and df["temporal_std"].dtype != object and not df.columns.str.endswith("_mean").any():
                # Could be a per-candidate summary where column is temporal_std (single value): treat as mean
                df["temporal_std_mean"] = df["temporal_std"]
                df["temporal_std_std"] = df.get("temporal_std_std", 0.0)
        # wwi
        if "wwi_percent_mean" not in df.columns and "wwi_percent" in df.columns:
            df["wwi_percent_mean"] = df["wwi_percent"]
            df["wwi_percent_std"] = df.get("wwi_percent_std", 0.0)
        # jaccard
        if "jaccard_with_original_mean" not in df.columns and "jaccard_with_original" in df.columns:
            df["jaccard_with_original_mean"] = df["jaccard_with_original"]
            df["jaccard_with_original_std"] = df.get("jaccard_with_original_std", 0.0)
        # If a single-row aggregated summary is present and parameter columns are missing, it is a per-run summary; allow it to pass through
        return df

    # Otherwise we assume a per-iteration table: aggregate by parameter set
    group_cols = [c for c in ["alpha", "beta", "gamma", "min_distance_km"] if c in df.columns]
    if not group_cols:
        raise ValueError("Cannot find parameter columns (alpha/beta/gamma/min_distance_km) in bootstrap summary")

    agg_map = {}
    # possible metric columns to aggregate
    if "temporal_std" in df.columns:
        agg_map["temporal_std"] = ["mean", "std"]
    if "wwi_percent" in df.columns:
        agg_map["wwi_percent"] = ["mean", "std"]
    if "jaccard_with_original" in df.columns:
        agg_map["jaccard_with_original"] = ["mean", "std"]

    if not agg_map:
        # Last resort: find numeric columns and compute mean/std
        metric_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in group_cols]
        for c in metric_cols:
            agg_map[c] = ["mean", "std"]

    grouped = df.groupby(group_cols).agg(agg_map).reset_index()
    # Flatten MultiIndex columns
    grouped.columns = ["_" . join([str(i) for i in col]).strip("_") if isinstance(col, tuple) else col for col in grouped.columns]
    # Normalize to expected names
    # e.g. temporal_std_mean -> temporal_std_mean
    return grouped.rename(columns=lambda x: x.replace("_mean", "_mean").replace("_std", "_std"))


def find_best_bootstrap_candidate(df: pd.DataFrame) -> dict:
    """Find best bootstrap candidate based on composite score.

    Criteria:
    - Lowest temporal_std_std (most stable)
    - Lowest wwi_percent_mean (best temporal diversity)
    - Highest jaccard_with_original_mean (most reproducible)
    """
    if len(df) == 0:
        raise ValueError("Empty bootstrap summary")

    try:
        df_norm = _normalize_and_aggregate_bootstrap_df(df)
    except Exception as e:
        raise ValueError(f"Could not normalize/aggregate bootstrap summary: {e}")

    # Ensure required columns exist
    required = ["temporal_std_mean", "temporal_std_std", "wwi_percent_mean", "wwi_percent_std", "jaccard_with_original_mean", "jaccard_with_original_std"]
    missing = [r for r in required if r not in df_norm.columns]
    if missing:
        raise ValueError(f"Bootstrap summary missing required aggregated columns: {missing}")

    df = df_norm.copy()
    # Normalize scores (0-1)
    df["stability_score"] = 1 - (
        df["temporal_std_std"] - df["temporal_std_std"].min()
    ) / (df["temporal_std_std"].max() - df["temporal_std_std"].min() + 1e-9)
    df["diversity_score"] = 1 - (
        df["wwi_percent_mean"] - df["wwi_percent_mean"].min()
    ) / (df["wwi_percent_mean"].max() - df["wwi_percent_mean"].min() + 1e-9)
    df["reproducibility_score"] = (df["jaccard_with_original_mean"] - df["jaccard_with_original_mean"].min()) / (
        df["jaccard_with_original_mean"].max() - df["jaccard_with_original_mean"].min() + 1e-9
    )

    # Composite: 40% stability + 30% diversity + 30% reproducibility
    df["composite_score"] = (
        0.4 * df["stability_score"] + 0.3 * df["diversity_score"] + 0.3 * df["reproducibility_score"]
    )

    best_idx = df["composite_score"].idxmax()
    best_row = df.loc[best_idx]

    # Return with friendly keys (keep older key names for backward compatibility)
    return {
        "alpha": float(best_row.get("alpha", 0.0)),
        "beta": float(best_row.get("beta", 0.0)),
        "gamma": float(best_row.get("gamma", 0.0)),
        "min_distance_km": float(best_row.get("min_distance_km", 0)),
        "composite_score": float(best_row["composite_score"]),
        "temporal_std_mean": float(best_row["temporal_std_mean"]),
        "wwi_percent_mean": float(best_row["wwi_percent_mean"]),
        "jaccard_mean": float(best_row["jaccard_with_original_mean"]),
    }


def inject_into_config(cfg_path: Path, params: dict, backup: bool = True):
    """Inject bootstrap-best params into pipeline config."""
    bak = cfg_path.with_suffix(cfg_path.suffix + ".bootstrap_bak")
    if backup:
        shutil.copy2(cfg_path, bak)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("selection", {})
    cfg["selection"]["alpha_visual"] = params["alpha"]
    cfg["selection"]["beta_spatial"] = params["beta"]
    cfg["selection"]["gamma_temporal"] = params["gamma"]
    cfg["selection"]["min_distance_km"] = params["min_distance_km"]
    # Add provenance
    cfg["selection"]["_bootstrap_provenance"] = {
        "composite_score": params["composite_score"],
        "temporal_std_mean": params["temporal_std_mean"],
        "wwi_percent_mean": params["wwi_percent_mean"],
        "jaccard_mean": params["jaccard_mean"],
    }

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return bak if backup else None


def write_new_config(
    out_path: Path,
    params: dict,
    base_cfg_path: Path = Path("config/pipeline_config.yaml"),
):
    """Write new config file with bootstrap-best params."""
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("selection", {})
    cfg["selection"]["alpha_visual"] = params["alpha"]
    cfg["selection"]["beta_spatial"] = params["beta"]
    cfg["selection"]["gamma_temporal"] = params["gamma"]
    cfg["selection"]["min_distance_km"] = params["min_distance_km"]
    cfg["selection"]["_bootstrap_provenance"] = {
        "composite_score": params["composite_score"],
        "temporal_std_mean": params["temporal_std_mean"],
        "wwi_percent_mean": params["wwi_percent_mean"],
        "jaccard_mean": params["jaccard_mean"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-summary", required=True, help="Path to bootstrap_summary.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--inject",
        action="store_true",
        help="Inject into config/pipeline_config.yaml (makes backup)",
    )
    group.add_argument(
        "--write-config", help="Write a separate YAML config with injected values"
    )

    args = parser.parse_args(argv)

    bootstrap_csv = Path(args.bootstrap_summary)
    if not bootstrap_csv.exists():
        print(f"Bootstrap summary not found: {bootstrap_csv}")
        return 1

    df = pd.read_csv(bootstrap_csv)

    # If there are no parameter columns in the CSV, try to augment from run-level best_trial.json
    param_cols = set(["alpha", "beta", "gamma", "min_distance_km"]) & set(df.columns)
    if not param_cols:
        # Attempt to find best_trial.json in the same run results directory
        run_results_dir = bootstrap_csv.parent
        candidate_paths = [run_results_dir / "best_trial.json", run_results_dir.parent / "results" / "best_trial.json"]
        best_trial = None
        for p in candidate_paths:
            if p.exists():
                try:
                    best_trial = json.loads(p.read_text())
                    break
                except Exception:
                    best_trial = None
        # Fallback: search all runs for the most recent best_trial.json
        if best_trial is None:
            runs_dir = Path("outputs") / "runs"
            if runs_dir.exists():
                candidates = list(runs_dir.glob("**/results/best_trial.json"))
                if candidates:
                    # pick most recently modified
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    for p in candidates:
                        try:
                            best_trial = json.loads(p.read_text())
                            break
                        except Exception:
                            continue
        if best_trial:
            # Inject parameters into the dataframe rows
            for k in ["a", "alpha"]:
                if k in best_trial:
                    df["alpha"] = float(best_trial.get(k))
                    break
            for k in ["b", "beta"]:
                if k in best_trial:
                    df["beta"] = float(best_trial.get(k))
                    break
            for k in ["c", "gamma"]:
                if k in best_trial:
                    df["gamma"] = float(best_trial.get(k))
                    break
            for k in ["min_distance_km", "min_distance"]:
                if k in best_trial:
                    df["min_distance_km"] = float(best_trial.get(k))
                    break
        else:
            # If CSV looks like a single-run summary (common case), try to write a config using autoscale or pipeline defaults
            if len(df) == 1:
                # Try autoscale best JSON
                autoscale_json = Path("outputs") / "optuna_autoscale_best_latest.json"
                autoscale_cfg = {}
                if autoscale_json.exists():
                    try:
                        autoscale_cfg = json.loads(autoscale_json.read_text()).get("user_attrs", {})
                    except Exception:
                        autoscale_cfg = {}
                if autoscale_cfg:
                    # Use autoscale as fallback params
                    params = {
                        "alpha": float(autoscale_cfg.get("alpha", 0.33)),
                        "beta": float(autoscale_cfg.get("beta", 0.33)),
                        "gamma": float(autoscale_cfg.get("gamma", 0.34)),
                        "min_distance_km": float(autoscale_cfg.get("min_distance_km", 50)),
                    }
                    df["alpha"] = params["alpha"]
                    df["beta"] = params["beta"]
                    df["gamma"] = params["gamma"]
                    df["min_distance_km"] = params["min_distance_km"]
                else:
                    # Last resort: use pipeline defaults from config/pipeline_config.yaml
                    import yaml

                    cfg_path = Path("config") / "pipeline_config.yaml"
                    if cfg_path.exists():
                        cfg = yaml.safe_load(cfg_path.read_text())
                        sel = cfg.get("selection", {})
                        df["alpha"] = sel.get("alpha_visual", 0.33)
                        df["beta"] = sel.get("beta_spatial", 0.33)
                        df["gamma"] = sel.get("gamma_temporal", 0.34)
                        df["min_distance_km"] = sel.get("min_distance_km", 50)
                    else:
                        raise ValueError(
                            f"Could not find parameter provenance (best_trial.json) for bootstrap summary {bootstrap_csv}, and neither autoscale nor pipeline config provide defaults."
                        )
    best = find_best_bootstrap_candidate(df)

    print(f"\n{'='*60}")
    print("BEST BOOTSTRAP CANDIDATE")
    print(f"{'='*60}")
    print(f"Composite Score:  {best['composite_score']:.4f}")
    print(f"Alpha (visual):   {best['alpha']:.2f}")
    print(f"Beta (spatial):   {best['beta']:.2f}")
    print(f"Gamma (temporal): {best['gamma']:.2f}")
    print(f"Min Distance:     {best['min_distance_km']:.0f} km")
    print("\nExpected Metrics:")
    print(f"  Temporal STD:   {best['temporal_std_mean']:.2f} years")
    print(f"  WWI Fraction:   {best['wwi_percent_mean']:.1f}%")
    print(f"  Jaccard (stab): {best['jaccard_mean']:.3f}")
    print(f"{'='*60}\n")

    if args.inject:
        cfg_path = Path("config/pipeline_config.yaml")
        bak = inject_into_config(cfg_path, best, backup=True)
        print(f"✅ Injected params into {cfg_path}")
        print(f"📁 Backup saved as {bak}")
    else:
        out = Path(args.write_config)
        write_new_config(out, best)
        print(f"✅ Wrote new config with injected params to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
