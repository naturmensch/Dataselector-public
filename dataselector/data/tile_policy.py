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
    flagged_count: int
    flagged_indices: list[int]
    flagged_shortnames: list[str]
    flagged_classes: list[str]
    flagged_caveats: list[dict[str, Any]]
    policy_path: str | None
    policy_sha256: str | None
    applied: bool


def load_tile_exclusion_policy(
    path: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
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


def _resolve_policy_ref(constants: dict[str, Any], ref: Any) -> Any:
    """Resolve dotted constant references from policy payloads."""
    if not isinstance(ref, str):
        return ref
    current: Any = constants
    for part in ref.split("."):
        key = str(part).strip()
        if not key or not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _rule_mask(
    df: pd.DataFrame,
    match: dict[str, Any],
    *,
    constants: dict[str, Any] | None = None,
) -> pd.Series:
    """Build row mask for one rule match block."""
    mask = pd.Series(True, index=df.index)
    if not isinstance(match, dict):
        return pd.Series(False, index=df.index)
    constant_values = dict(constants or {})

    def _numeric_mask(column: str, operator: str, raw_value: Any) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        series = pd.to_numeric(df[column], errors="coerce")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return pd.Series(False, index=df.index)
        if operator == "lt":
            local = series < value
        elif operator == "lte":
            local = series <= value
        elif operator == "gt":
            local = series > value
        elif operator == "gte":
            local = series >= value
        else:
            return pd.Series(False, index=df.index)
        return local.fillna(False)

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

    for value_key, ref_key, operator in (
        ("year_lt", "year_lt_ref", "lt"),
        ("year_lte", "year_lte_ref", "lte"),
        ("year_gt", "year_gt_ref", "gt"),
        ("year_gte", "year_gte_ref", "gte"),
    ):
        has_direct = value_key in match
        has_ref = ref_key in match
        if not has_direct and not has_ref:
            continue
        raw_value = match.get(value_key)
        if raw_value is None and has_ref:
            raw_value = _resolve_policy_ref(constant_values, match.get(ref_key))
        mask = mask & _numeric_mask("year", operator, raw_value)

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
    constants = payload.get("constants", {})
    if not isinstance(constants, dict):
        raise ValueError("tile exclusion policy 'constants' must be a mapping")
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("tile exclusion policy 'rules' must be a list")

    if len(rules) == 0:
        return TilePolicyResult(
            dataframe=df,
            excluded_count=0,
            excluded_indices=[],
            excluded_shortnames=[],
            flagged_count=0,
            flagged_indices=[],
            flagged_shortnames=[],
            flagged_classes=[],
            flagged_caveats=[],
            policy_path=str(path_obj) if path_obj else None,
            policy_sha256=(
                compute_file_sha256(path_obj)
                if path_obj and path_obj.exists()
                else None
            ),
            applied=False,
        )

    exclude_mask = pd.Series(False, index=df.index)
    flagged_records: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", "")).strip().lower()
        match = rule.get("match", {})
        rule_mask = _rule_mask(df, match, constants=constants)
        if action == "exclude_from_candidate_pool":
            exclude_mask = exclude_mask | rule_mask
            continue
        if action != "flag_for_reporting":
            continue
        matched_df = df.loc[rule_mask]
        rule_id = str(rule.get("id", "")).strip()
        rule_class = str(rule.get("class", "")).strip()
        rationale = str(rule.get("rationale", "")).strip()
        for idx, row in matched_df.iterrows():
            record: dict[str, Any] = {
                "index": int(idx),
                "rule_id": rule_id,
                "class": rule_class,
                "rationale": rationale,
            }
            for column in ("shortName", "longName", "city"):
                if column in matched_df.columns:
                    value = row.get(column)
                    if pd.notna(value):
                        record[column] = str(value)
            if "year" in matched_df.columns:
                year_value = row.get("year")
                if pd.notna(year_value):
                    try:
                        record["year"] = int(float(year_value))
                    except Exception:
                        record["year"] = str(year_value)
            flagged_records.append(record)

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

    flagged_indices = sorted(
        {int(record["index"]) for record in flagged_records if "index" in record}
    )
    flagged_shortnames = sorted(
        {
            str(record["shortName"])
            for record in flagged_records
            if record.get("shortName") is not None
        }
    )
    flagged_classes = sorted(
        {
            str(record["class"])
            for record in flagged_records
            if record.get("class") is not None
        }
    )

    return TilePolicyResult(
        dataframe=filtered,
        excluded_count=int(exclude_mask.sum()),
        excluded_indices=[int(i) for i in excluded_df.index.tolist()],
        excluded_shortnames=[str(v) for v in excluded_shortnames],
        flagged_count=len(flagged_records),
        flagged_indices=flagged_indices,
        flagged_shortnames=flagged_shortnames,
        flagged_classes=flagged_classes,
        flagged_caveats=flagged_records,
        policy_path=str(path_obj) if path_obj else None,
        policy_sha256=(
            compute_file_sha256(path_obj) if path_obj and path_obj.exists() else None
        ),
        applied=True,
    )
