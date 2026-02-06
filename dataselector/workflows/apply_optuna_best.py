"""Apply best Optuna trial parameters to pipeline configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml

from dataselector.cli_decorators import cli_command


def find_best_trial(df: pd.DataFrame) -> pd.Series:
    """Return the best trial row (maximum value)."""
    if "value" not in df.columns:
        raise ValueError("Optuna results CSV has no 'value' column")
    return df.loc[df["value"].idxmax()]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column name by suffix."""
    for col in df.columns:
        for cand in candidates:
            if col.endswith(cand):
                return col
    return None


def extract_params_from_trial(df: pd.DataFrame, best_row: pd.Series) -> dict:
    """Extract normalized alpha/beta/gamma and min_distance_km from best row."""
    alpha_col = find_col(
        df,
        [
            "user_attrs_alpha",
            "user_attrs_a",
            "user_attrs_alpha_value",
            "params_alpha",
            "alpha",
            "a",
        ],
    )
    beta_col = find_col(df, ["user_attrs_beta", "user_attrs_b", "params_beta", "beta", "b"])
    gamma_col = find_col(
        df,
        ["user_attrs_gamma", "user_attrs_c", "params_gamma", "gamma", "c"],
    )
    min_col = find_col(
        df,
        [
            "user_attrs_min_distance_km",
            "user_attrs_min_distance",
            "params_min_distance_km",
            "params_min_distance",
            "min_distance_km",
        ],
    )

    def safe_get(col: str | None):
        if col and col in best_row.index and pd.notna(best_row[col]):
            return best_row[col]
        return None

    alpha_raw = safe_get(alpha_col)
    beta_raw = safe_get(beta_col)
    gamma_raw = safe_get(gamma_col)
    min_distance = safe_get(min_col)

    alpha = beta = gamma = None
    if alpha_raw is not None and beta_raw is not None and gamma_raw is not None:
        try:
            a = float(alpha_raw)
            b = float(beta_raw)
            g = float(gamma_raw)
            total = a + b + g
            if total > 0:
                alpha = a / total
                beta = b / total
                gamma = g / total
        except Exception:
            alpha = beta = gamma = None

    if min_distance is not None:
        try:
            min_distance = int(float(min_distance))
        except Exception:
            min_distance = None

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "min_distance_km": min_distance,
    }


def _apply_selection(cfg: dict, params: dict) -> dict:
    cfg.setdefault("selection", {})
    if params.get("alpha") is not None:
        cfg["selection"]["alpha_visual"] = float(params["alpha"])
    if params.get("beta") is not None:
        cfg["selection"]["beta_spatial"] = float(params["beta"])
    if params.get("gamma") is not None:
        cfg["selection"]["gamma_temporal"] = float(params["gamma"])
    if params.get("min_distance_km") is not None:
        cfg["selection"]["min_distance_km"] = float(params["min_distance_km"])
    return cfg


def inject_into_config(cfg_path: Path, params: dict, backup: bool = True) -> Path | None:
    """Inject params into an existing YAML config file."""
    bak = cfg_path.with_suffix(cfg_path.suffix + ".optuna_bak")
    if backup:
        shutil.copy2(cfg_path, bak)

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = _apply_selection(cfg, params)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return bak if backup else None


def write_new_config(
    out_path: Path,
    params: dict,
    base_cfg_path: Path = Path("config/pipeline_config.yaml"),
) -> Path:
    """Write a derived config file from base config + Optuna params."""
    cfg = yaml.safe_load(base_cfg_path.read_text()) or {}
    cfg = _apply_selection(cfg, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


@cli_command(
    "apply-optuna-best",
    help="Apply best Optuna trial to pipeline config",
    args={
        "optuna_csv": {
            "type": str,
            "required": True,
            "help": "Path to optuna results CSV",
        },
        "inject": {
            "type": bool,
            "action": "store_true",
            "help": "Inject into config/pipeline_config.yaml with backup",
        },
        "write_config": {
            "type": str,
            "default": None,
            "help": "Write a derived config to this path",
        },
    },
)
def main(
    optuna_csv: str,
    inject: bool = False,
    write_config: str | None = None,
) -> int:
    """CLI entry point for applying best Optuna trial params."""
    if inject == (write_config is not None):
        print("Choose exactly one mode: --inject or --write-config <path>")
        return 1

    csv_path = Path(optuna_csv)
    if not csv_path.exists():
        print(f"Optuna CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    best = find_best_trial(df)
    params = extract_params_from_trial(df, best)
    print(f"Best trial params: {params}")

    if inject:
        cfg_path = Path("config/pipeline_config.yaml")
        bak = inject_into_config(cfg_path, params, backup=True)
        print(f"Injected params into {cfg_path}; backup saved as {bak}")
    else:
        out_path = Path(write_config)
        write_new_config(out_path, params)
        print(f"Wrote new config with injected params to: {out_path}")

    return 0
