#!/usr/bin/env python3
"""Apply best Bootstrap candidate into pipeline config.

Usage:
  python scripts/apply_bootstrap_best.py --bootstrap-summary outputs/fine_sweep/bootstrap_summary.csv --inject
  python scripts/apply_bootstrap_best.py --bootstrap-summary outputs/bootstrap_results_summary.csv --write-config config/pipeline_config.bootstrap.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import shutil


def find_best_bootstrap_candidate(df: pd.DataFrame) -> dict:
    """Find best bootstrap candidate based on composite score.
    
    Criteria:
    - Lowest temporal_std_std (most stable)
    - Lowest wwi_percent_mean (best temporal diversity)
    - Highest jaccard_mean (most reproducible)
    """
    if len(df) == 0:
        raise ValueError("Empty bootstrap summary")
    
    # Normalize scores (0-1)
    df = df.copy()
    df['stability_score'] = 1 - (df['temporal_std_std'] - df['temporal_std_std'].min()) / (df['temporal_std_std'].max() - df['temporal_std_std'].min() + 1e-9)
    df['diversity_score'] = 1 - (df['wwi_percent_mean'] - df['wwi_percent_mean'].min()) / (df['wwi_percent_mean'].max() - df['wwi_percent_mean'].min() + 1e-9)
    df['reproducibility_score'] = (df['jaccard_mean'] - df['jaccard_mean'].min()) / (df['jaccard_mean'].max() - df['jaccard_mean'].min() + 1e-9)
    
    # Composite: 40% stability + 30% diversity + 30% reproducibility
    df['composite_score'] = 0.4 * df['stability_score'] + 0.3 * df['diversity_score'] + 0.3 * df['reproducibility_score']
    
    best_idx = df['composite_score'].idxmax()
    best_row = df.loc[best_idx]
    
    return {
        'alpha': float(best_row['alpha']),
        'beta': float(best_row['beta']),
        'gamma': float(best_row['gamma']),
        'min_distance_km': float(best_row['min_distance_km']),
        'composite_score': float(best_row['composite_score']),
        'temporal_std_mean': float(best_row['temporal_std_mean']),
        'wwi_percent_mean': float(best_row['wwi_percent_mean']),
        'jaccard_mean': float(best_row['jaccard_mean']),
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


def write_new_config(out_path: Path, params: dict, base_cfg_path: Path = Path("config/pipeline_config.yaml")):
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
    group.add_argument("--inject", action="store_true", help="Inject into config/pipeline_config.yaml (makes backup)")
    group.add_argument("--write-config", help="Write a separate YAML config with injected values")
    
    args = parser.parse_args(argv)
    
    bootstrap_csv = Path(args.bootstrap_summary)
    if not bootstrap_csv.exists():
        print(f"Bootstrap summary not found: {bootstrap_csv}")
        return 1
    
    df = pd.read_csv(bootstrap_csv)
    best = find_best_bootstrap_candidate(df)
    
    print(f"\n{'='*60}")
    print("BEST BOOTSTRAP CANDIDATE")
    print(f"{'='*60}")
    print(f"Composite Score:  {best['composite_score']:.4f}")
    print(f"Alpha (visual):   {best['alpha']:.2f}")
    print(f"Beta (spatial):   {best['beta']:.2f}")
    print(f"Gamma (temporal): {best['gamma']:.2f}")
    print(f"Min Distance:     {best['min_distance_km']:.0f} km")
    print(f"\nExpected Metrics:")
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
