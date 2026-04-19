from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .measure_state import WidthCalibrationSession
from .models import (
    DEFAULT_DISPLAY_CROP_FACTOR,
    DEFAULT_DISPLAY_SCALE,
    REJECTION_REASONS,
    VIEWER_HOTKEY_HELP,
    TaskRecord,
    normalize_source_fid,
)
from .prepare import pixel_window


def display_crop_size_px(task_crop_size_px: int, display_crop_factor: float) -> int:
    factor = float(display_crop_factor)
    if factor <= 0.0 or factor > 1.0:
        raise ValueError("display_crop_factor must be in the range (0, 1].")
    size = int(round(float(task_crop_size_px) * factor))
    size = max(16, size)
    size = min(int(task_crop_size_px), size)
    if size % 2 != int(task_crop_size_px) % 2:
        size = min(int(task_crop_size_px), size + 1)
    return int(size)


def upsample_nearest_rgb(image: np.ndarray, scale: int) -> np.ndarray:
    factor = int(scale)
    if factor <= 1:
        return image
    up = np.repeat(image, factor, axis=0)
    return np.repeat(up, factor, axis=1)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def select_interactive_matplotlib_backend(matplotlib_module: Any) -> str:
    current_backend = str(matplotlib_module.get_backend())
    current_lower = current_backend.lower()
    if "qt" in current_lower and "agg" in current_lower:
        return current_backend
    if not (has_module("PySide6") and has_module("matplotlib.backends.backend_qtagg")):
        raise RuntimeError(
            "measure-width-calibration requires a Qt 6 backend. "
            "PySide6 and matplotlib QtAgg must be available."
        )
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    try:
        matplotlib_module.use("QtAgg", force=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to activate the Qt 6 matplotlib backend (QtAgg) for "
            "measure-width-calibration."
        ) from exc
    return "QtAgg"


def ui_scale_from_screen_metrics(
    screen_dpi: float | None,
    *,
    screen_width_px: float | None = None,
    screen_height_px: float | None = None,
) -> float:
    dpi_scale = 1.0
    if (
        screen_dpi is not None
        and np.isfinite(float(screen_dpi))
        and float(screen_dpi) > 0.0
    ):
        dpi_scale = float(screen_dpi) / 96.0
    resolution_scale = 1.0
    if (
        screen_width_px is not None
        and screen_height_px is not None
        and float(screen_width_px) > 0.0
        and float(screen_height_px) > 0.0
    ):
        resolution_scale = max(
            float(screen_width_px) / 1920.0, float(screen_height_px) / 1080.0
        )
    return float(min(2.5, max(1.0, dpi_scale, resolution_scale)))


class InteractiveMeasurementViewer:
    def __init__(
        self,
        *,
        handoff_dir: Path,
        session: WidthCalibrationSession,
        display_crop_factor: float = DEFAULT_DISPLAY_CROP_FACTOR,
        display_scale: int = DEFAULT_DISPLAY_SCALE,
    ) -> None:
        self.handoff_dir = handoff_dir
        self.session = session
        self.display_crop_factor = float(display_crop_factor)
        self.display_scale = max(1, int(display_scale))
        self.fig = None
        self.ax = None
        self.plt = None
        self.current_task: TaskRecord | None = None
        self.current_crop: np.ndarray | None = None
        self.current_window: tuple[int, int, int, int] | None = None
        self.current_clicks: list[tuple[float, float]] = []
        self._window = None
        self._ui_scale = 1.0
        self._qt_control_bar = None
        self._qt_status_bar = None

    def run(self) -> dict[str, Any]:
        import matplotlib

        select_interactive_matplotlib_backend(matplotlib)
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._configure_window_ui()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        print("Interactive width calibration")
        print("Left-click twice to measure width.")
        print(
            f"Display settings: crop_factor={self.display_crop_factor:.2f}, scale={self.display_scale}x"
        )
        print(VIEWER_HOTKEY_HELP)
        self._show_next_task()
        plt.show()
        return self.session.summary()

    def _configure_window_ui(self) -> None:
        manager = getattr(self.fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        toolbar = getattr(manager, "toolbar", None)
        if window is None:
            return
        self._window = window
        if not self._configure_qt_window_ui(window, toolbar):
            raise RuntimeError(
                "measure-width-calibration requires a Qt 6 window manager. "
                "The current matplotlib backend window is not Qt-based."
            )

    def _configure_qt_window_ui(self, window: Any, toolbar: Any) -> bool:
        try:
            from PySide6 import QtCore, QtWidgets
        except Exception:
            return False
        if not isinstance(window, QtWidgets.QMainWindow):
            return False
        self._ui_scale = self._estimate_qt_ui_scale(window)
        self._apply_qt_window_start_mode(window)
        if toolbar is not None:
            try:
                toolbar.hide()
            except Exception:
                pass
            try:
                window.removeToolBar(toolbar)
            except Exception:
                pass
        existing_bar = window.findChild(
            QtWidgets.QToolBar, "WidthCalibrationControlsBar"
        )
        if existing_bar is not None:
            try:
                window.removeToolBar(existing_bar)
            except Exception:
                pass
            existing_bar.deleteLater()
        bar = QtWidgets.QToolBar("Calibration Controls", window)
        bar.setObjectName("WidthCalibrationControlsBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        button_font_px = max(13, int(round(13 * self._ui_scale)))
        label_font_px = max(12, int(round(12 * self._ui_scale)))
        button_min_height = max(32, int(round(32 * self._ui_scale)))
        button_min_width = max(84, int(round(84 * self._ui_scale)))
        button_padding_y = max(6, int(round(6 * self._ui_scale)))
        button_padding_x = max(10, int(round(10 * self._ui_scale)))
        gap_px = max(6, int(round(6 * self._ui_scale)))
        margin_px = max(6, int(round(6 * self._ui_scale)))
        bar.setStyleSheet(
            "\n".join(
                [
                    "QToolBar {"
                    f" spacing: {gap_px}px;"
                    f" padding: {margin_px}px;"
                    "}",
                    "QToolButton, QPushButton {"
                    f" font-size: {button_font_px}px;"
                    " font-weight: 600;"
                    f" min-height: {button_min_height}px;"
                    f" min-width: {button_min_width}px;"
                    f" padding: {button_padding_y}px {button_padding_x}px;"
                    "}",
                    "QLabel {"
                    f" font-size: {label_font_px}px;"
                    f" padding-left: {gap_px * 2}px;"
                    "}",
                ]
            )
        )

        def add_button(label: str, callback: Any) -> None:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            bar.addWidget(button)

        add_button("Clear Clicks", self._clear_clicks)
        add_button("Reject", self._reject_current_task)
        add_button("Skip", self._skip_current_task)
        add_button("Undo", self._undo_last_task)
        add_button("Quit", self._quit_viewer)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        bar.addWidget(spacer)
        help_label = QtWidgets.QLabel(VIEWER_HOTKEY_HELP)
        help_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        bar.addWidget(help_label)
        self._qt_control_bar = bar
        window.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, bar)
        status_bar = window.statusBar()
        self._qt_status_bar = status_bar
        status_bar.setSizeGripEnabled(False)
        status_bar.setStyleSheet(
            "QStatusBar {"
            f" font-size: {label_font_px}px;"
            f" min-height: {max(24, int(round(24 * self._ui_scale)))}px;"
            "}"
        )
        status_bar.showMessage(VIEWER_HOTKEY_HELP)
        try:
            bar.show()
            bar.raise_()
            status_bar.show()
            window.update()
            window.repaint()
        except Exception:
            pass
        return True

    def _estimate_qt_ui_scale(self, window: Any) -> float:
        return 1.0

    def _apply_qt_window_start_mode(self, window: Any) -> None:
        try:
            window.showNormal()
        except Exception:
            pass
        try:
            window.showMaximized()
        except Exception:
            pass
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass

    def _clear_clicks(self) -> None:
        self.current_clicks = []
        if self.current_task is not None:
            self._draw_current()

    def _skip_current_task(self) -> None:
        if self.current_task is None:
            return
        self.session.defer_task(self.current_task.task_id)
        self._show_next_task()

    def _undo_last_task(self) -> None:
        undone = self.session.undo_last()
        print(
            "Nothing to undo."
            if undone is None
            else f"Undid measurement for task {undone.get('task_id', '')}"
        )
        self._show_next_task()

    def _quit_viewer(self) -> None:
        if self.plt is not None:
            self.plt.close(self.fig)

    def _task_object_reference_text(self, task: TaskRecord | None) -> str:
        if task is None:
            return ""
        source_fid = normalize_source_fid(task.source_fid)
        if source_fid:
            return f"fid={source_fid}"
        if str(task.source_feature_id).strip():
            return f"source={task.source_feature_id}"
        return ""

    def _update_status_message(self) -> None:
        if self._qt_status_bar is None or self.current_task is None:
            return
        reference = self._task_object_reference_text(self.current_task)
        progress = self.session.progress_snapshot(self.current_task.task_id)
        round_segment = (
            f"round={progress['current_round_index']}/{progress['current_round_total']}"
            if int(progress.get("current_round_total", 0)) > 0
            else f"round={self.current_task.pass_type}"
        )
        class_pending = int(
            progress.get("pending_by_class", {}).get(int(self.current_task.class_id), 0)
        )
        prefix_parts = [
            part
            for part in (
                reference,
                f"class={self.current_task.class_id}",
                f"patch={self.current_task.patch_id}",
                f"pass={self.current_task.pass_type}",
                round_segment,
                f"pos={progress['current_position']}/{progress['eligible_total']}",
                f"remaining={progress['current_remaining_total']}",
                f"remaining_pass={progress['current_remaining_in_pass']}",
                f"remaining_class={class_pending}",
            )
            if str(part).strip()
        ]
        self._qt_status_bar.showMessage(
            f"{' | '.join(prefix_parts)} | {VIEWER_HOTKEY_HELP}"
        )

    def _show_reject_dialog(self) -> tuple[str, str] | None:
        if self._window is None or self.current_task is None:
            return None
        try:
            from PySide6 import QtCore, QtWidgets
        except Exception as exc:
            raise RuntimeError(
                "Reject dialog requires PySide6/Qt6 to be available."
            ) from exc
        dialog = QtWidgets.QDialog(self._window)
        dialog.setWindowTitle(f"Reject {self.current_task.task_id}")
        dialog.setModal(True)
        dialog.setMinimumWidth(460)
        layout = QtWidgets.QVBoxLayout(dialog)
        reference = self._task_object_reference_text(self.current_task)
        info_text = " | ".join(
            part
            for part in (
                reference,
                f"class={self.current_task.class_id}",
                f"patch={self.current_task.patch_id}",
                f"pass={self.current_task.pass_type}",
            )
            if str(part).strip()
        )
        info_label = QtWidgets.QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        layout.addWidget(QtWidgets.QLabel("Reject reason"))
        reason_combo = QtWidgets.QComboBox(dialog)
        reason_combo.addItem("Choose reject reason...", "")
        for reason in REJECTION_REASONS:
            reason_combo.addItem(reason, reason)
        layout.addWidget(reason_combo)
        layout.addWidget(QtWidgets.QLabel("Note (optional)"))
        note_edit = QtWidgets.QPlainTextEdit(dialog)
        note_edit.setPlaceholderText("Optional note")
        note_edit.setFixedHeight(96)
        layout.addWidget(note_edit)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            dialog,
        )
        reject_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        reject_button.setText("Reject")
        reject_button.setEnabled(False)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        reason_combo.currentIndexChanged.connect(
            lambda _idx: reject_button.setEnabled(bool(reason_combo.currentData()))
        )
        layout.addWidget(button_box)
        reason_combo.setFocus()
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None
        reason = str(reason_combo.currentData() or "").strip()
        if reason not in REJECTION_REASONS:
            return None
        return reason, note_edit.toPlainText().strip()

    def _reject_current_task(self) -> None:
        if self.current_task is None:
            return
        dialog_result = self._show_reject_dialog()
        if dialog_result is None:
            return
        reason, note = dialog_result
        self.session.record_reject(
            self.current_task.task_id, reject_reason=reason, note=note
        )
        self._show_next_task()

    def _show_next_task(self) -> None:
        task = self.session.next_task()
        if task is None:
            assert self.plt is not None
            self.ax.clear()
            self.ax.set_title("All eligible tasks completed")
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            self.plt.close(self.fig)
            return
        self.current_task = task
        self.current_clicks = []
        self.current_crop, self.current_window = self._load_crop(task)
        self._draw_current()
        if self._window is not None:
            try:
                if self._qt_control_bar is not None:
                    self._qt_control_bar.show()
                    self._qt_control_bar.raise_()
                self._window.update()
                self._window.repaint()
            except Exception:
                pass

    def _load_crop(
        self, task: TaskRecord
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        quicklook_path = (self.handoff_dir / task.quicklook_path).resolve()
        with rasterio.open(quicklook_path) as ds:
            arr = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
        window = pixel_window(
            anchor_x_px=task.anchor_x_px,
            anchor_y_px=task.anchor_y_px,
            crop_size_px=display_crop_size_px(
                task.crop_size_px, self.display_crop_factor
            ),
            width=arr.shape[1],
            height=arr.shape[0],
        )
        if window is None:
            raise ValueError(f"Task crop is out of bounds: {task}")
        x0, y0, x1, y1 = window
        return arr[y0:y1, x0:x1], window

    def _draw_current(self) -> None:
        assert self.current_task is not None
        assert self.current_crop is not None
        assert self.current_window is not None
        x0, y0, _x1, _y1 = self.current_window
        display_image = upsample_nearest_rgb(self.current_crop, self.display_scale)
        anchor_x_local = (self.current_task.anchor_x_px - x0) * self.display_scale
        anchor_y_local = (self.current_task.anchor_y_px - y0) * self.display_scale
        self.ax.clear()
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.95)
        self.ax.imshow(display_image, interpolation="nearest")
        self.ax.scatter(
            [anchor_x_local], [anchor_y_local], c="yellow", s=50, marker="+"
        )
        for click in self.current_clicks:
            self.ax.scatter(
                [(click[0] - x0) * self.display_scale],
                [(click[1] - y0) * self.display_scale],
                c="red",
                s=36,
            )
        reference = self._task_object_reference_text(self.current_task)
        reference_segment = f" | {reference}" if reference else ""
        progress = self.session.progress_snapshot(self.current_task.task_id)
        round_segment = (
            f"round={progress['current_round_index']}/{progress['current_round_total']}"
            if int(progress.get("current_round_total", 0)) > 0
            else f"round={self.current_task.pass_type}"
        )
        class_pending = int(
            progress.get("pending_by_class", {}).get(int(self.current_task.class_id), 0)
        )
        self.ax.set_title(
            f"{self.current_task.task_id} | class={self.current_task.class_id}"
            f"{reference_segment} | patch={self.current_task.patch_id}"
            f" | pass={self.current_task.pass_type} | {round_segment}"
            f" | pos={progress['current_position']}/{progress['eligible_total']}"
            f" | remaining={progress['current_remaining_total']}"
            f" (pass={progress['current_remaining_in_pass']}, class={class_pending})"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()
        self._update_status_message()
        if self._window is not None:
            try:
                self._window.update()
                self._window.repaint()
            except Exception:
                pass

    def _on_click(self, event: Any) -> None:
        if self.current_task is None or self.current_window is None:
            return
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or event.button != 1
        ):
            return
        x0, y0, _x1, _y1 = self.current_window
        abs_x = float(x0 + (float(event.xdata) / float(self.display_scale)))
        abs_y = float(y0 + (float(event.ydata) / float(self.display_scale)))
        self.current_clicks.append((abs_x, abs_y))
        if len(self.current_clicks) >= 2:
            click1, click2 = self.current_clicks[:2]
            self.session.record_accept(
                self.current_task.task_id, click1=click1, click2=click2
            )
            self._show_next_task()
            return
        self._draw_current()

    def _on_key(self, event: Any) -> None:
        if self.current_task is None:
            return
        key = str(event.key).lower()
        if key == "escape":
            self._clear_clicks()
        elif key == "s":
            self._skip_current_task()
        elif key == "u":
            self._undo_last_task()
        elif key == "q":
            self._quit_viewer()
        elif key == "r":
            self._reject_current_task()


__all__ = [
    "InteractiveMeasurementViewer",
    "display_crop_size_px",
    "has_module",
    "select_interactive_matplotlib_backend",
    "ui_scale_from_screen_metrics",
    "upsample_nearest_rgb",
]
