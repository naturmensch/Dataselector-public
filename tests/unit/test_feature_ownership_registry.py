from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "status" / "feature_ownership_registry.yaml"


def _collect_cli_owners() -> dict[str, list[str]]:
    owners: dict[str, set[str]] = {}
    for path in ROOT.joinpath("dataselector").rglob("*.py"):
        if path.name == "cli_decorators.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "cli_command"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                owners.setdefault(first.value, set()).add(str(path.relative_to(ROOT)))
    return {cmd: sorted(paths) for cmd, paths in owners.items()}


def test_feature_ownership_registry_exists_and_has_commands():
    assert REGISTRY.exists()
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "commands" in data
    assert isinstance(data["commands"], dict)
    assert data["commands"]


def test_registry_command_owners_exist_in_repo():
    commands = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["commands"]
    for cmd, entry in commands.items():
        owner = entry.get("canonical_owner_module")
        assert owner, f"{cmd}: missing canonical_owner_module"
        assert (ROOT / owner).exists(), f"{cmd}: owner path missing: {owner}"


def test_registry_matches_detected_cli_owners():
    commands = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["commands"]
    detected = _collect_cli_owners()

    for cmd, owners in detected.items():
        assert cmd in commands, f"CLI command missing in registry: {cmd}"
        assert len(owners) == 1, f"{cmd}: expected exactly one owner, got {owners}"
        assert commands[cmd]["canonical_owner_module"] == owners[0]
