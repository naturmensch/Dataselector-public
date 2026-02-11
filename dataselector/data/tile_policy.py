"""Tile-level exclusion policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dataselector.runtime.parameter_snapshot import compute_file_sha256


@dataclass(frozen=True)
class TilePolicyResult:
    """Outcome of applying a tile exclusion policy."""

    dataframe: pd.DataFrame
    excluded_count: int
    excluded_indices: list[int]
    excluded_shortnames: list[str]
    policy_path: str | None
    policy_sha256: str | None
    applied: bool


def load_tile_exclusion_policy(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    """Load tile exclusion policy YAML if present."""
    if path is None:
        return {}, None
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Tile exclusion policy not found: {policy_path}")
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tile exclusion policy format: {policy_path}")
    return payload, policy_path


def _rule_mask(df: pd.DataFrame, match: dict[str, Any]) -> pd.Series:
    """Build row mask for one rule match block."""
    mask = pd.Series(True, index=df.index)
    if not isinstance(match, dict):
        return pd.Series(False, index=df.index)

    if "shortName" in match:
        if "shortName" not in df.columns:
            return pd.Series(False, index=df.index)
        values = match["shortName"]
        if not isinstance(values, list):
            values = [values]
        wanted = {str(v).strip().lower() for v in values}
        mask = mask & df["shortName"].astype(str).str.strip().str.lower().isin(wanted)

    if "longName_contains" in match:
        if "longName" not in df.columns:
            return pd.Series(False, index=df.index)
        values = match["longName_contains"]
        if not isinstance(values, list):
            values = [values]
        local = pd.Series(False, index=df.index)
        for value in values:
            needle = str(value).strip().lower()
            if not needle:
                continue
            local = local | df["longName"].astype(str).str.lower().str.contains(needle)
        mask = mask & local

    return mask


def apply_tile_exclusion_policy(
    df: pd.DataFrame,
    *,
    policy: dict[str, Any] | None = None,
    policy_path: str | Path | None = None,
) -> TilePolicyResult:
    """Apply exclude-from-candidate-pool rules and return filtered dataframe."""
    payload = dict(policy or {})
    path_obj: Path | None = Path(policy_path) if policy_path else None
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("tile exclusion policy 'rules' must be a list")

    if len(rules) == 0:
        return TilePolicyResult(
            dataframe=df,
            excluded_count=0,
            excluded_indices=[],
            excluded_shortnames=[],
            policy_path=str(path_obj) if path_obj else None,
            policy_sha256=compute_file_sha256(path_obj) if path_obj and path_obj.exists() else None,
            applied=False,
        )

    exclude_mask = pd.Series(False, index=df.index)
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", "")).strip().lower()
        if action != "exclude_from_candidate_pool":
            continue
        match = rule.get("match", {})
        exclude_mask = exclude_mask | _rule_mask(df, match)

    excluded_df = df.loc[exclude_mask]
    filtered = df.loc[~exclude_mask].copy()
    excluded_shortnames = []
    if "shortName" in excluded_df.columns:
        excluded_shortnames = (
            excluded_df["shortName"].astype(str).dropna().unique().tolist()
        )
    elif "longName" in excluded_df.columns:
        excluded_shortnames = (
            excluded_df["longName"].astype(str).dropna().unique().tolist()
        )

    return TilePolicyResult(
        dataframe=filtered,
        excluded_count=int(exclude_mask.sum()),
        excluded_indices=[int(i) for i in excluded_df.index.tolist()],
        excluded_shortnames=[str(v) for v in excluded_shortnames],
        policy_path=str(path_obj) if path_obj else None,
        policy_sha256=compute_file_sha256(path_obj) if path_obj and path_obj.exists() else None,
        applied=True,
    )
