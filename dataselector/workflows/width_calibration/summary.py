from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataselector.runtime.parameter_snapshot import compute_file_sha256

from .measure_state import load_measurements_csv
from .models import (
    SUMMARY_COLUMNS,
    SUMMARY_FILENAME,
    SUMMARY_JSON_FILENAME,
    WORKFLOW_VERSION,
    SummaryRow,
    normalize_class,
    repo_root,
    resolve_path,
    utc_now,
    write_json,
)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(quantile(values, 0.75) - quantile(values, 0.25))


def mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def bool_from_keep(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_width_calibration(
    *,
    measurements_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    repo_root_path = repo_root()
    measurements_path = resolve_path(
        measurements_csv, repo_root_path=repo_root_path, prefer_repo=False
    ).resolve()
    out_dir_path = resolve_path(
        out_dir, repo_root_path=repo_root_path, prefer_repo=True
    ).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    measurements_df = load_measurements_csv(measurements_path)
    if measurements_df.empty:
        raise ValueError(f"No measurements available: {measurements_path}")
    measurements_df = measurements_df.copy()
    measurements_df["class"] = measurements_df["class"].map(normalize_class)
    measurements_df["keep_bool"] = measurements_df["keep"].map(bool_from_keep)
    measurements_df["width_px_num"] = pd.to_numeric(
        measurements_df["width_px"], errors="coerce"
    )
    primary_valid = measurements_df.loc[
        (measurements_df["pass_type"] == "primary")
        & (measurements_df["keep_bool"])
        & measurements_df["width_px_num"].notna()
    ].copy()
    repeat_valid = measurements_df.loc[
        (measurements_df["pass_type"] == "repeat")
        & (measurements_df["keep_bool"])
        & measurements_df["width_px_num"].notna()
    ].copy()
    primary_by_task = primary_valid[["task_id", "width_px_num"]].rename(
        columns={"task_id": "primary_task_id", "width_px_num": "primary_width_px"}
    )
    repeat_pairs = repeat_valid.merge(
        primary_by_task,
        left_on="repeat_of_task_id",
        right_on="primary_task_id",
        how="left",
    )
    repeat_pairs = repeat_pairs.loc[repeat_pairs["primary_width_px"].notna()].copy()
    repeat_pairs["abs_diff_px"] = (
        repeat_pairs["width_px_num"] - repeat_pairs["primary_width_px"]
    ).abs()
    present_classes = sorted(
        int(class_id)
        for class_id in primary_valid["class"].dropna().astype(int).unique().tolist()
    )
    summary_rows: list[dict[str, Any]] = []
    for class_id in present_classes:
        class_primary = primary_valid.loc[primary_valid["class"] == class_id].copy()
        values = class_primary["width_px_num"].astype(float).to_numpy()
        class_repeat_pairs = repeat_pairs.loc[repeat_pairs["class"] == class_id].copy()
        repeat_abs = class_repeat_pairs["abs_diff_px"].astype(float).to_numpy()
        median_px = float(np.median(values)) if values.size else float("nan")
        summary_rows.append(
            SummaryRow(
                class_id=int(class_id),
                n_valid_primary=int(values.size),
                median_px=round(median_px, 6) if values.size else np.nan,
                IQR_px=round(iqr(values), 6) if values.size else np.nan,
                MAD_px=round(mad(values), 6) if values.size else np.nan,
                repeat_median_abs_diff_px=(
                    round(float(np.median(repeat_abs)), 6)
                    if repeat_abs.size
                    else np.nan
                ),
                low_evidence_flag=bool(values.size < 5),
                high_variance_flag=bool(iqr(values) > 2.0) if values.size else False,
                low_reliability_flag=(
                    bool(float(np.median(repeat_abs)) > 1.0)
                    if repeat_abs.size
                    else False
                ),
                final_width_px=int(round(median_px)) if values.size else np.nan,
            ).to_row()
        )
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = out_dir_path / SUMMARY_FILENAME
    summary_df.to_csv(summary_path, index=False)
    summary_json_path = out_dir_path / SUMMARY_JSON_FILENAME
    summary_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "generated_utc": utc_now(),
        "measurements_csv": str(measurements_path),
        "measurements_csv_sha256": compute_file_sha256(measurements_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": compute_file_sha256(summary_path),
        "present_classes": present_classes,
        "results": summary_df.to_dict("records"),
    }
    write_json(summary_json_path, summary_payload)
    return {
        "summary_csv": str(summary_path),
        "summary_json": str(summary_json_path),
        "present_classes": present_classes,
    }


__all__ = ["summarize_width_calibration", "bool_from_keep", "iqr", "mad", "quantile"]
