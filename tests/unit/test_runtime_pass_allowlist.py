import ast
import json
from pathlib import Path

ALLOWLIST_PATH = Path("docs/status/runtime_pass_allowlist_2026-02-08.json")
RUNTIME_ROOT = Path("dataselector")


def _nearest_non_empty(lines: list[str], start: int, step: int) -> str:
    idx = start
    while 0 <= idx < len(lines):
        value = lines[idx].strip()
        if value:
            return value
        idx += step
    return "<none>"


def _pass_fingerprint(lines: list[str], lineno: int) -> str:
    # Fingerprint by nearest non-empty context around `pass`.
    before = _nearest_non_empty(lines, lineno - 2, -1)
    after = _nearest_non_empty(lines, lineno, 1)
    return f"{before}|pass|{after}"


def _collect_runtime_passes() -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    for py_file in RUNTIME_ROOT.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(py_file))
        rel = str(py_file).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                entries.add((rel, _pass_fingerprint(lines, node.lineno)))
    return entries


def _load_allowlist() -> set[tuple[str, str]]:
    payload = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    out: set[tuple[str, str]] = set()
    for item in payload.get("entries", []):
        fingerprint = item.get("fingerprint")
        if not fingerprint:
            path = Path(item["file"])
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            fingerprint = _pass_fingerprint(lines, int(item["line"]))
        out.add((item["file"], fingerprint))
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
