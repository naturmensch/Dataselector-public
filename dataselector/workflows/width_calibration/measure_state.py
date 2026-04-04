from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    DEFAULT_DISPLAY_CROP_FACTOR,
    DEFAULT_DISPLAY_SCALE,
    LEGACY_MEASUREMENT_REQUIRED_COLUMNS,
    LEGACY_TASK_REQUIRED_COLUMNS,
    MEASUREMENTS_FILENAME,
    MEASUREMENT_COLUMNS,
    MeasurementRecord,
    REJECTION_REASONS,
    TASK_COLUMNS,
    TaskRecord,
    normalize_source_fid,
    repo_root,
    require_columns,
    resolve_path,
)
from .prepare import source_fid_lookup_from_roads
from .runs import default_roads_gpkg_path, preflight_measure_width_calibration


def load_tasks_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Task CSV not found: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    require_columns(df, LEGACY_TASK_REQUIRED_COLUMNS, "width_calibration_tasks.csv")
    if "source_fid" not in df.columns:
        df["source_fid"] = ""
    df["source_fid"] = df["source_fid"].map(normalize_source_fid)
    missing_fid = (
        df["source_fid"].astype(str).str.strip().eq("")
        & df["source_feature_id"].astype(str).str.strip().ne("")
    )
    if missing_fid.any():
        lookup = source_fid_lookup_from_roads(default_roads_gpkg_path(repo_root_path=repo_root()))
        if lookup:
            df.loc[missing_fid, "source_fid"] = df.loc[missing_fid, "source_feature_id"].map(
                lambda value: lookup.get(str(value).strip(), "")
            )
    for col in ("class", "anchor_x_px", "anchor_y_px", "crop_size_px", "queue_position"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    pass_order = {"primary": 0, "repeat": 1}
    df["_pass_order"] = df["pass_type"].astype(str).map(pass_order).fillna(99).astype(int)
    df = df.sort_values(["_pass_order", "class", "queue_position", "task_id"], kind="stable").reset_index(drop=True)
    return df.drop(columns=["_pass_order"])[TASK_COLUMNS].copy()


def empty_measurements_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MEASUREMENT_COLUMNS)


def load_measurements_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_measurements_df()
    df = pd.read_csv(path, dtype=str).fillna("")
    require_columns(df, LEGACY_MEASUREMENT_REQUIRED_COLUMNS, MEASUREMENTS_FILENAME)
    if "source_fid" not in df.columns:
        df["source_fid"] = ""
    df["source_fid"] = df["source_fid"].map(normalize_source_fid)
    return df[MEASUREMENT_COLUMNS].copy()


def ensure_measurements_csv_schema(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    changed = False
    for column in MEASUREMENT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
            changed = True
    ordered = df[MEASUREMENT_COLUMNS].copy()
    if changed or list(df.columns) != MEASUREMENT_COLUMNS:
        ordered.to_csv(path, index=False)


def next_measure_id(measurements_df: pd.DataFrame) -> str:
    max_idx = 0
    for value in measurements_df["measure_id"].astype(str).tolist():
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            max_idx = max(max_idx, int(digits))
    return f"measure_{max_idx + 1:05d}"


def append_measurement_row(path: Path, row: MeasurementRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        ensure_measurements_csv_schema(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.to_row().get(col, "") for col in MEASUREMENT_COLUMNS})


def undo_last_measurement(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    last = rows.pop()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MEASUREMENT_COLUMNS})
    return {str(k): str(v) for k, v in last.items()}


def measurement_keep_map(measurements_df: pd.DataFrame) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in measurements_df.to_dict("records"):
        keep_text = str(row.get("keep", "")).strip().lower()
        out[str(row["task_id"])] = keep_text in {"1", "true", "yes"}
    return out


class WidthCalibrationSession:
    def __init__(self, *, tasks_df: pd.DataFrame, measurements_path: Path):
        self.tasks_df = tasks_df.reset_index(drop=True).copy()
        self.measurements_path = measurements_path
        self._tasks = [TaskRecord.from_mapping(row) for row in self.tasks_df.to_dict("records")]
        self._task_index = {task.task_id: task for task in self._tasks}
        self._deferred_task_ids: list[str] = []
        self._priority_task_id: str | None = None
        self.reload()

    def reload(self) -> None:
        self.measurements_df = load_measurements_csv(self.measurements_path)

    def pending_task_ids(self) -> list[str]:
        completed = set(self.measurements_df["task_id"].astype(str).tolist())
        keep_map = measurement_keep_map(self.measurements_df)
        regular: list[str] = []
        deferred: list[str] = []
        for task in self._tasks:
            if task.task_id in completed:
                continue
            if task.pass_type == "repeat" and not keep_map.get(task.repeat_of_task_id, False):
                continue
            if task.task_id in self._deferred_task_ids:
                deferred.append(task.task_id)
            else:
                regular.append(task.task_id)
        ordered = regular + [task_id for task_id in deferred if task_id not in regular]
        if self._priority_task_id and self._priority_task_id in ordered:
            ordered = [self._priority_task_id] + [task_id for task_id in ordered if task_id != self._priority_task_id]
        return ordered

    def next_task(self) -> TaskRecord | None:
        pending = self.pending_task_ids()
        if not pending:
            return None
        return self._task_index[pending[0]]

    def defer_task(self, task_id: str) -> None:
        if task_id not in self._deferred_task_ids:
            self._deferred_task_ids.append(task_id)
        if self._priority_task_id == task_id:
            self._priority_task_id = None

    def record_accept(
        self,
        task_id: str,
        *,
        click1: tuple[float, float],
        click2: tuple[float, float],
        note: str = "",
    ) -> dict[str, Any]:
        task = self._task_index[str(task_id)]
        width_px = float(np.hypot(float(click1[0]) - float(click2[0]), float(click1[1]) - float(click2[1])))
        row = MeasurementRecord(
            task_id=task.task_id,
            candidate_id=task.candidate_id,
            class_id=task.class_id,
            patch_id=task.patch_id,
            tile_shortname=task.tile_shortname,
            source_fid=normalize_source_fid(task.source_fid),
            source_feature_id=task.source_feature_id,
            measure_id=next_measure_id(self.measurements_df),
            pass_type=task.pass_type,
            repeat_of_task_id=task.repeat_of_task_id,
            anchor_x_px=task.anchor_x_px,
            anchor_y_px=task.anchor_y_px,
            click1_x_px=round(float(click1[0]), 6),
            click1_y_px=round(float(click1[1]), 6),
            click2_x_px=round(float(click2[0]), 6),
            click2_y_px=round(float(click2[1]), 6),
            width_px=round(width_px, 6),
            keep=1,
            reject_reason="",
            note=str(note).strip(),
        )
        append_measurement_row(self.measurements_path, row)
        self.reload()
        self._deferred_task_ids = [item for item in self._deferred_task_ids if item != str(task_id)]
        self._priority_task_id = None
        return row.to_row()

    def record_reject(self, task_id: str, *, reject_reason: str, note: str = "") -> dict[str, Any]:
        if reject_reason not in REJECTION_REASONS:
            raise ValueError(f"Unsupported reject reason: {reject_reason!r}")
        task = self._task_index[str(task_id)]
        row = MeasurementRecord(
            task_id=task.task_id,
            candidate_id=task.candidate_id,
            class_id=task.class_id,
            patch_id=task.patch_id,
            tile_shortname=task.tile_shortname,
            source_fid=normalize_source_fid(task.source_fid),
            source_feature_id=task.source_feature_id,
            measure_id=next_measure_id(self.measurements_df),
            pass_type=task.pass_type,
            repeat_of_task_id=task.repeat_of_task_id,
            anchor_x_px=task.anchor_x_px,
            anchor_y_px=task.anchor_y_px,
            click1_x_px="",
            click1_y_px="",
            click2_x_px="",
            click2_y_px="",
            width_px="",
            keep=0,
            reject_reason=reject_reason,
            note=str(note).strip(),
        )
        append_measurement_row(self.measurements_path, row)
        self.reload()
        self._deferred_task_ids = [item for item in self._deferred_task_ids if item != str(task_id)]
        self._priority_task_id = None
        return row.to_row()

    def undo_last(self) -> dict[str, str] | None:
        undone = undo_last_measurement(self.measurements_path)
        self.reload()
        if undone is not None:
            task_id = str(undone.get("task_id", "")).strip()
            if task_id:
                self._priority_task_id = task_id
        return undone

    def summary(self) -> dict[str, Any]:
        pending = self.pending_task_ids()
        return {
            "measurements_csv": str(self.measurements_path),
            "recorded_count": int(len(self.measurements_df)),
            "pending_count": int(len(pending)),
            "pending_task_ids": pending,
        }


def measure_width_calibration(
    *,
    handoff_dir: str | Path,
    tasks_csv: str | Path,
    out_csv: str | Path,
    display_crop_factor: float = DEFAULT_DISPLAY_CROP_FACTOR,
    display_scale: int = DEFAULT_DISPLAY_SCALE,
    resume: bool = False,
    prompt_for_sync: bool = False,
) -> dict[str, Any]:
    from .viewer_qt import InteractiveMeasurementViewer

    repo_root_path = repo_root()
    handoff_dir_path = resolve_path(handoff_dir, repo_root_path=repo_root_path, prefer_repo=True).resolve()
    tasks_csv_path = resolve_path(tasks_csv, repo_root_path=repo_root_path, prefer_repo=True).resolve()
    out_csv_path = resolve_path(out_csv, repo_root_path=repo_root_path, prefer_repo=False).resolve()
    if out_csv_path.exists() and not bool(resume):
        raise FileExistsError(
            f"Measurement CSV already exists. Use --resume to continue: {out_csv_path}"
        )
    preflight_measure_width_calibration(
        tasks_csv_path=tasks_csv_path,
        repo_root_path=repo_root_path,
        prompt_for_sync=prompt_for_sync,
    )
    tasks_df = load_tasks_csv(tasks_csv_path)
    session = WidthCalibrationSession(tasks_df=tasks_df, measurements_path=out_csv_path)
    viewer = InteractiveMeasurementViewer(
        handoff_dir=handoff_dir_path,
        session=session,
        display_crop_factor=float(display_crop_factor),
        display_scale=int(display_scale),
    )
    return viewer.run()


__all__ = [
    "WidthCalibrationSession",
    "ensure_measurements_csv_schema",
    "load_measurements_csv",
    "load_tasks_csv",
    "measure_width_calibration",
    "measurement_keep_map",
]
