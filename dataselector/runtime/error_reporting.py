"""Centralized exception reporting helpers for runtime workflows."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _exc_info_tuple(
    exc: BaseException,
) -> tuple[type[BaseException], BaseException, Any]:
    return (type(exc), exc, exc.__traceback__)


def _append_exception_artifact(
    *,
    output_dir: Path,
    phase: str,
    context: dict[str, Any],
    traceback_text: str,
) -> Path:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = diagnostics_dir / "exceptions.log"

    lines = [
        "=" * 80,
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"phase: {phase}",
        "context:",
        json.dumps(_json_safe(context), ensure_ascii=True, indent=2, sort_keys=True),
        "traceback:",
        traceback_text.rstrip(),
        "",
    ]
    with artifact_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return artifact_path


def report_exception(
    exc: BaseException,
    *,
    phase: str,
    user_message: str,
    output_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
    context: dict[str, Any] | None = None,
    echo_traceback: bool = True,
) -> dict[str, Any]:
    """Emit a concise runtime error plus full traceback and structured metadata."""

    context_payload = _json_safe(context or {})
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "context": context_payload,
    }

    exceptions_log_path: str | None = None
    if output_dir is not None:
        artifact_path = _append_exception_artifact(
            output_dir=Path(output_dir),
            phase=phase,
            context=context_payload,
            traceback_text=traceback_text,
        )
        exceptions_log_path = str(artifact_path)
        record["exceptions_log_path"] = exceptions_log_path

    print(f"❌ {user_message}: {exc}")
    if echo_traceback:
        print(traceback_text.rstrip())

    if logger is not None:
        logger.error(
            "%s | phase=%s | context=%s",
            user_message,
            phase,
            json.dumps(context_payload, ensure_ascii=True, sort_keys=True),
            exc_info=_exc_info_tuple(exc),
        )

    return record


def log_expected_exception(
    logger: logging.Logger,
    message: str,
    *,
    exc: BaseException | None = None,
    context: dict[str, Any] | None = None,
    level: int = logging.WARNING,
) -> None:
    """Log non-fatal fallback exceptions with full context."""

    context_payload = json.dumps(
        _json_safe(context or {}),
        ensure_ascii=True,
        sort_keys=True,
    )
    if exc is None:
        logger.log(level, "%s | context=%s", message, context_payload)
        return
    logger.log(
        level,
        "%s | context=%s",
        message,
        context_payload,
        exc_info=_exc_info_tuple(exc),
    )
