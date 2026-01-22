#!/usr/bin/env python3
"""Apply best Optuna trial into pipeline config or write a new config file.

Usage:
  python scripts/apply_optuna_best.py --optuna-csv outputs/optuna_results.csv --write-config config/pipeline_config.optuna.yaml
  python scripts/apply_optuna_best.py --optuna-csv outputs/optuna_results.csv --inject  # overwrite with backup

This script extracts the best trial (max 'value') from the Optuna results CSV and
injects the derived normalized alpha/beta/gamma and min_distance into the
pipeline config. It can either inject into `config/pipeline_config.yaml` (with
backup) or write a new config file (`--write-config path`). The backup is
saved as `config/pipeline_config.yaml.optuna_bak` if `--inject` is used.

The script returns exit code 0 on success, 1 on failure.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml


def find_best_trial(df: pd.DataFrame):
    if "value" not in df.columns:
        raise ValueError("Optuna results CSV has no 'value' column")
    best = df.loc[df["value"].idxmax()]
    return best


def find_col(df, candidates):
    for c in df.columns:
        for cand in candidates:
            if c.endswith(cand):
                return c
    return None


def extract_params_from_trial(df: pd.DataFrame, best_row) -> dict:
    # try to find the user_attrs / params columns for alpha, beta, gamma, min_distance
    alpha_col = (
        find_col(
            df,
            [
                "user_attrs_alpha",
                "user_attrs_a",
                "user_attrs_alpha_value",
                "user_attrs_alpha",
            ],
        )
        or find_col(df, ["params_alpha", "a", "alpha"])
        or "alpha"
    )
    beta_col = (
        find_col(df, ["user_attrs_beta", "user_attrs_b"])
        or find_col(df, ["params_beta", "b", "beta"])
        or "beta"
    )
    gamma_col = (
        find_col(df, ["user_attrs_gamma", "user_attrs_c"])
        or find_col(df, ["params_gamma", "c", "gamma"])
        or "gamma"
    )
    min_col = find_col(
        df,
        [
            "user_attrs_min_distance_km",
            "user_attrs_min_distance",
            "params_min_distance_km",
            "params_min_distance",
        ],
    )

    def safe_get(col):
        if col and col in best_row.index and pd.notna(best_row[col]):
            return best_row[col]
        return None

    a = safe_get(alpha_col)
    b = safe_get(beta_col)
    g = safe_get(gamma_col)
    min_d = safe_get(min_col)

    # If raw a,b,g are present (unnormalized), normalize
    vals = [v for v in [a, b, g] if v is not None]
    if len(vals) == 3:
        try:
            a_f = float(a)
            b_f = float(b)
            g_f = float(g)
            s = a_f + b_f + g_f
            if s > 0:
                alpha = a_f / s
                beta = b_f / s
                gamma = g_f / s
            else:
                alpha = beta = gamma = None
        except Exception:
            alpha = beta = gamma = None
    else:
        alpha = safe_get("alpha") or safe_get("a") or None
        beta = safe_get("beta") or safe_get("b") or None
        gamma = safe_get("gamma") or safe_get("c") or None

    if min_d is not None:
        try:
            min_d = int(float(min_d))
        except Exception:
            min_d = None

    return {"alpha": alpha, "beta": beta, "gamma": gamma, "min_distance_km": min_d}


def inject_into_config(cfg_path: Path, params: dict, backup: bool = True):
    bak = cfg_path.with_suffix(cfg_path.suffix + ".optuna_bak")
    if backup:
        shutil.copy2(cfg_path, bak)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("selection", {})
    if params["alpha"] is not None:
        cfg["selection"]["alpha_visual"] = (
            float(params["alpha"])
            if params["alpha"] is not None
            else cfg["selection"].get("alpha_visual")
        )
    if params["beta"] is not None:
        cfg["selection"]["beta_spatial"] = (
            float(params["beta"])
            if params["beta"] is not None
            else cfg["selection"].get("beta_spatial")
        )
    if params["gamma"] is not None:
        cfg["selection"]["gamma_temporal"] = (
            float(params["gamma"])
            if params["gamma"] is not None
            else cfg["selection"].get("gamma_temporal")
        )
    if params["min_distance_km"] is not None:
        cfg["selection"]["min_distance_km"] = (
            float(params["min_distance_km"])
            if params["min_distance_km"] is not None
            else cfg["selection"].get("min_distance_km")
        )
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return bak if backup else None


def write_new_config(
    out_path: Path,
    params: dict,
    base_cfg_path: Path = Path("config/pipeline_config.yaml"),
):
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("selection", {})
    if params["alpha"] is not None:
        cfg["selection"]["alpha_visual"] = float(params["alpha"])
    if params["beta"] is not None:
        cfg["selection"]["beta_spatial"] = float(params["beta"])
    if params["gamma"] is not None:
        cfg["selection"]["gamma_temporal"] = float(params["gamma"])
    if params["min_distance_km"] is not None:
        cfg["selection"]["min_distance_km"] = float(params["min_distance_km"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optuna-csv", required=True, help="Path to optuna_results.csv"
    )
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

    optuna_csv = Path(args.optuna_csv)
    if not optuna_csv.exists():
        print(f"Optuna CSV not found: {optuna_csv}")
        return 1

    df = pd.read_csv(optuna_csv)
    best = find_best_trial(df)
    params = extract_params_from_trial(df, best)
    # basic validation: alpha/beta/gamma plausibility
    # If alpha not normalized but provided as floats between 0 and 1, keep them
    print(f"Best trial params: {params}")

    if args.inject:
        cfg_path = Path("config/pipeline_config.yaml")
        bak = inject_into_config(cfg_path, params, backup=True)
        print(f"Injected params into {cfg_path}; backup saved as {bak}")
    else:
        out = Path(args.write_config)
        write_new_config(out, params)
        print(f"Wrote new config with injected params to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
