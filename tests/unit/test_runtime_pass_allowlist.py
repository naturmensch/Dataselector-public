import ast
import json
from pathlib import Path

ALLOWLIST_PATH = Path("docs/status/runtime_pass_allowlist_2026-02-08.json")
RUNTIME_ROOT = Path("dataselector")


def _collect_runtime_passes() -> set[tuple[str, int]]:
    entries: set[tuple[str, int]] = set()
    for py_file in RUNTIME_ROOT.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        rel = str(py_file).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                entries.add((rel, node.lineno))
    return entries


def _load_allowlist() -> set[tuple[str, int]]:
    payload = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    out: set[tuple[str, int]] = set()
    for item in payload.get("entries", []):
        out.add((item["file"], int(item["line"])))
    return out


def test_runtime_pass_allowlist_is_complete_and_strict():
    assert ALLOWLIST_PATH.exists(), "runtime pass allowlist missing"

    current = _collect_runtime_passes()
    allowed = _load_allowlist()

    missing_in_allowlist = sorted(current - allowed)
    stale_allowlist_entries = sorted(allowed - current)

    assert not missing_in_allowlist, (
        "New runtime `pass` statements detected. Classify and document them in "
        f"{ALLOWLIST_PATH}: {missing_in_allowlist}"
    )
    assert not stale_allowlist_entries, (
        "Allowlist contains stale entries. Remove obsolete records from "
        f"{ALLOWLIST_PATH}: {stale_allowlist_entries}"
    )
