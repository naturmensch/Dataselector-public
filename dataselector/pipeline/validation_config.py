"""Validation config helpers.

Loads pipeline config without defaults and provides strict accessors.
All required values must be present in the YAML; missing values raise errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when required config values are missing."""


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")
    return data


def _dig(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def get_required(config: dict, paths: Iterable[str], label: str) -> Any:
    for path in paths:
        value = _dig(config, path)
        if value is not None:
            return value
    tried = ", ".join(paths)
    raise ConfigError(f"Missing required config value for {label}. Tried: {tried}")


def get_optional(config: dict, paths: Iterable[str]) -> Any:
    for path in paths:
        value = _dig(config, path)
        if value is not None:
            return value
    return None
