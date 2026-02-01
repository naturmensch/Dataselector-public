import os
import tempfile
from pathlib import Path
import pandas as pd
import yaml

from scripts.apply_optuna_best import extract_params_from_trial, find_best_trial, write_new_config, inject_into_config


def make_dummy_optuna_csv(path: Path):
    # Create a small dataframe emulating optuna trials dataframe
    df = pd.DataFrame([
        {"value": 0.5, "user_attrs_alpha": 0.7, "user_attrs_beta": 0.1, "user_attrs_gamma": 0.2, "user_attrs_min_distance_km": 50},
        {"value": 0.8, "user_attrs_alpha": 0.6, "user_attrs_beta": 0.15, "user_attrs_gamma": 0.25, "user_attrs_min_distance_km": 37},
    ])
    df.to_csv(path, index=False)
    return df


def test_extract_and_write_config(tmp_path):
    optuna_csv = tmp_path / "optuna_results.csv"
    make_dummy_optuna_csv(optuna_csv)
    df = pd.read_csv(optuna_csv)
    best = find_best_trial(df)
    params = extract_params_from_trial(df, best)

    assert params["alpha"] is not None
    assert 0 <= params["alpha"] <= 1
    assert params["min_distance_km"] == 37

    # write new config
    out_cfg = tmp_path / "pipeline_config.optuna.yaml"
    write_new_config(out_cfg, params, base_cfg_path=Path("config/pipeline_config.yaml"))
    assert out_cfg.exists()
    cfg = yaml.safe_load(out_cfg.read_text())
    sel = cfg.get("selection", {})
    assert float(sel.get("alpha_visual")) == float(params["alpha"])
    assert int(sel.get("min_distance_km")) == int(params["min_distance_km"])


def test_inject_and_backup(tmp_path, monkeypatch):
    # Copy original config to tmp and inject into that copy
    cfg_path = tmp_path / "pipeline_config.yaml"
    orig_cfg = yaml.safe_load(Path("config/pipeline_config.yaml").read_text())
    cfg_path.write_text(yaml.safe_dump(orig_cfg))

    optuna_csv = tmp_path / "optuna_results.csv"
    make_dummy_optuna_csv(optuna_csv)
    df = pd.read_csv(optuna_csv)
    best = find_best_trial(df)
    params = extract_params_from_trial(df, best)

    bak = inject_into_config(cfg_path, params, backup=True)
    assert bak.exists()
    new_cfg = yaml.safe_load(cfg_path.read_text())
    sel = new_cfg.get("selection", {})
    assert float(sel.get("alpha_visual")) == float(params["alpha"])

