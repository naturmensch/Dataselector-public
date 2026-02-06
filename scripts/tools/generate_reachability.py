#!/usr/bin/env python3
"""Generate reachability matrix for scripts/ based on static string matches.

Outputs:
 - outputs/reachability_matrix.csv
 - outputs/reachability_report.md

Heuristics used:
 ./scripts/exec_in_env.sh --env dataselector -- - match invocations like: python scripts/X.py
 - match exec_in_env.sh ... python scripts/X.py
 - match "-m scripts.X" (module invocation)
 - match imports: import scripts.X or from scripts.X import ...
 - mark doc-only references separately when matches found only in docs/ or README

This is a static approximation (no runtime tracing).
"""

from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patterns
PY_SCRIPT_PAT = re.compile(
    r"\bpython\b[^\n]*?\b(?:\.\/)?scripts\/([A-Za-z0-9_\-]+)\.py\b"
)
EXEC_IN_ENV_PAT = re.compile(
    r"exec_in_env\.sh[^\n]*?\bpython\b[^\n]*?scripts\/([A-Za-z0-9_\-]+)\.py\b"
)
MODULE_PAT = re.compile(r"-m\s+scripts\.([A-Za-z0-9_]+)\b")
IMPORT_PAT = re.compile(r"\b(?:from|import)\s+scripts\.([A-Za-z0-9_]+)\b")
SCRIPT_FILE_REF_PAT = re.compile(r"scripts\/([A-Za-z0-9_\-]+)\.py")

# Files to scan (restrict to likely callers)
SCAN_ROOTS = [ROOT / "scripts", ROOT / "tests", ROOT / "docs", ROOT]
EXCLUDE_DIRS = {".git", "data", "outputs"}

# Collect edges: caller -> {callee: [evidence lines]}
edges = defaultdict(lambda: defaultdict(list))
# Also record doc-only refs
doc_refs = defaultdict(list)
# Keep set of discovered node names (basename .py)
all_scripts = {p.name for p in SCRIPTS_DIR.glob("*.py")}

for root in SCAN_ROOTS:
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in fp.parts):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception:
            continue

        # AST-based detection for Python files: detect variables referencing script paths and subprocess/os.system calls
        if fp.suffix == ".py":
            try:
                tree = ast.parse(text)
            except Exception:
                tree = None
            if tree is not None:
                # map var names to script filenames when assigned a string containing scripts/
                var_to_script = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        consts = [
                            n.value
                            for n in ast.walk(node.value)
                            if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        ]
                        # try 'scripts/xxx.py' form first
                        found_script = None
                        for c in consts:
                            if "scripts/" in c:
                                m = SCRIPT_FILE_REF_PAT.search(c)
                                if m:
                                    found_script = m.group(1) + ".py"
                                    break
                        # catch Path-like constructions: ROOT / 'scripts' / 'file.py'
                        if not found_script:
                            for c in consts:
                                if c.endswith(".py"):
                                    found_script = Path(c).name
                                    break
                        # aliasing: RHS might be a Name that references a var we already resolved
                        if not found_script:
                            for n in ast.walk(node.value):
                                if isinstance(n, ast.Name) and n.id in var_to_script:
                                    found_script = var_to_script[n.id]
                                    break
                        if found_script:
                            for t in node.targets:
                                if isinstance(t, ast.Name):
                                    var_to_script[t.id] = found_script
                # second pass: resolve aliases where RHS references names that we already mapped (fixes ternary and aliasing cases)
                changed = True
                while changed:
                    changed = False
                    for node2 in ast.walk(tree):
                        if isinstance(node2, ast.Assign):
                            for t in node2.targets:
                                if (
                                    isinstance(t, ast.Name)
                                    and t.id not in var_to_script
                                ):
                                    for n2 in ast.walk(node2.value):
                                        if (
                                            isinstance(n2, ast.Name)
                                            and n2.id in var_to_script
                                        ):
                                            var_to_script[t.id] = var_to_script[n2.id]
                                            changed = True
                                            break
                # find JoinedStr (f-strings) anywhere and detect formatted vars (e.g., str(MAIN_SCRIPT) used in command strings)
                for js in [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]:
                    found = set()
                    for part in js.values:
                        if isinstance(part, ast.FormattedValue):
                            if isinstance(part.value, ast.Name):
                                var = part.value.id
                                if var in var_to_script:
                                    found.add(var_to_script[var])
                            else:
                                for n2 in ast.walk(part.value):
                                    if (
                                        isinstance(n2, ast.Name)
                                        and n2.id in var_to_script
                                    ):
                                        found.add(var_to_script[n2.id])
                                        break
                    for sc in found:
                        edges[fp.name][sc].append(
                            (fp.as_posix(), getattr(js, "lineno", 0), "AST-joinedstr")
                        )

                # find subprocess/os.system calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # check args for constants or lists containing script refs
                        found = set()
                        # positional args
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(
                                arg.value, str
                            ):
                                s = arg.value
                                m = SCRIPT_FILE_REF_PAT.search(s)
                                if m:
                                    found.add(m.group(1) + ".py")
                                mm = re.search(r"-m\s+scripts\.([A-Za-z0-9_]+)", s)
                                if mm:
                                    found.add(mm.group(1) + ".py")
                            if isinstance(arg, (ast.List, ast.Tuple)):
                                values = [
                                    e.value
                                    for e in arg.elts
                                    if isinstance(e, ast.Constant)
                                    and isinstance(e.value, str)
                                ]
                                joined = " ".join(map(str, values))
                                m = SCRIPT_FILE_REF_PAT.search(joined)
                                if m:
                                    found.add(m.group(1) + ".py")
                                mm = re.search(r"-m\s+scripts\.([A-Za-z0-9_]+)", joined)
                                if mm:
                                    found.add(mm.group(1) + ".py")
                            if isinstance(arg, ast.JoinedStr):
                                # f-string: check formatted values referring to vars (handles str(MAIN_SCRIPT) and direct var usage)
                                for part in arg.values:
                                    if isinstance(part, ast.FormattedValue):
                                        # direct name
                                        if isinstance(part.value, ast.Name):
                                            var = part.value.id
                                            if var in var_to_script:
                                                found.add(var_to_script[var])
                                        else:
                                            # call or expression that might contain a Name referring to a script var
                                            for n in ast.walk(part.value):
                                                if (
                                                    isinstance(n, ast.Name)
                                                    and n.id in var_to_script
                                                ):
                                                    found.add(var_to_script[n.id])
                                                    break
                        # keyword args
                        for kw in node.keywords:
                            v = kw.value
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                s = v.value
                                m = SCRIPT_FILE_REF_PAT.search(s)
                                if m:
                                    found.add(m.group(1) + ".py")
                        # record any found scripts
                        for sc in found:
                            edges[fp.name][sc].append(
                                (
                                    fp.as_posix(),
                                    getattr(node, "lineno", 0),
                                    f'AST-call: {ast.unparse(node.func) if hasattr(ast, "unparse") else "call"}',
                                )
                            )

        for i, line in enumerate(text.splitlines(), start=1):
            # find python scripts
            for m in PY_SCRIPT_PAT.finditer(line):
                callee = m.group(1) + ".py"
                edges[fp.name][callee].append((fp.as_posix(), i, line.strip()))
            for m in EXEC_IN_ENV_PAT.finditer(line):
                callee = m.group(1) + ".py"
                edges[fp.name][callee].append((fp.as_posix(), i, line.strip()))
            for m in MODULE_PAT.finditer(line):
                callee = m.group(1) + ".py"
                edges[fp.name][callee].append((fp.as_posix(), i, line.strip()))
            for m in IMPORT_PAT.finditer(line):
                callee = m.group(1) + ".py"
                edges[fp.name][callee].append((fp.as_posix(), i, line.strip()))
            # generic script file refs (docs often reference these)
            for m in SCRIPT_FILE_REF_PAT.finditer(line):
                callee = m.group(1) + ".py"
                # if caller is in docs/ or README, mark as doc ref
                if (
                    "docs" in fp.parts
                    or fp.name.lower().startswith("readme")
                    or "README" in fp.name
                ):
                    doc_refs[callee].append((fp.as_posix(), i, line.strip()))
                else:
                    # also add as edge (some scripts include direct references)
                    edges[fp.name][callee].append((fp.as_posix(), i, line.strip()))

# Normalize detected callee names to actual files in scripts/ when possible (e.g., base names -> _modern/_OLD variants)
for caller in list(edges):
    for callee in list(edges[caller]):
        if callee not in all_scripts:
            base = callee.rstrip(".py")
            candidate = None
            for a in all_scripts:
                if a.startswith(base) or base in a:
                    candidate = a
                    break
            if candidate:
                edges[caller][candidate].extend(edges[caller][callee])
                del edges[caller][callee]

# Build adjacency graph (only for known scripts)
graph = defaultdict(set)
for caller, callees in edges.items():
    for callee in callees:
        if callee in all_scripts:
            graph[caller].add(callee)

# Entry points (start nodes) -- use basenames to match keys used for callers
ENTRY_TEST = "test_e2e_thesis_pipeline.py"
ENTRY_MONITOR = "xxl_full_run_monitor.py"
ENTRY_SH = "run_complete_thesis_pipeline.sh"
entry_nodes = [ENTRY_TEST, ENTRY_MONITOR, ENTRY_SH]


# BFS from entries to compute reachability
def bfs(start_nodes):
    q = deque()
    seen = set()
    parent = {}
    for s in start_nodes:
        q.append(s)
        seen.add(s)
        parent[s] = None
    while q:
        cur = q.popleft()
        for nb in graph.get(cur, []):
            if nb not in seen:
                seen.add(nb)
                parent[nb] = cur
                q.append(nb)
    return seen, parent


seen, parent = bfs(entry_nodes)

# Prepare CSV rows for all scripts
rows = []
for script in sorted(all_scripts):
    reachable = script in seen
    # find shortest path from any entry
    path = []
    if reachable:
        # climb parents until we reach a start node
        node = script
        rev = [node]
        while node not in entry_nodes and node in parent and parent[node] is not None:
            node = parent[node]
            rev.append(node)
        rev = list(reversed(rev))
        path = rev
    # gather evidence: earliest caller->script evidence
    evidence = []
    # find any caller that references script
    for caller, callees in edges.items():
        if script in callees:
            for ev in callees[script]:
                evidence.append(f"{ev[0]}:{ev[1]}: {ev[2]}")
    # also doc refs
    if script in doc_refs and not evidence:
        for ev in doc_refs[script]:
            evidence.append(f"DOC:{ev[0]}:{ev[1]}: {ev[2]}")

    rows.append(
        {
            "script": script,
            "reachable": "yes" if reachable else "no",
            "shortest_path": " -> ".join(path) if path else "",
            "evidence_sample": evidence[0] if evidence else "",
            "num_callers_found": sum(1 for c in edges if script in edges[c]),
        }
    )

# write CSV
csv_out = OUT_DIR / "reachability_matrix.csv"
with open(csv_out, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "script",
            "reachable",
            "shortest_path",
            "evidence_sample",
            "num_callers_found",
        ],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# write short report
report = OUT_DIR / "reachability_report.md"
with open(report, "w") as fh:
    fh.write("# Reachability Report\n\n")
    fh.write("Entry points scanned:\n")
    for e in entry_nodes:
        fh.write(f"- {e}\n")
    fh.write("\n")
    fh.write("## Summary\n")
    n_total = len(all_scripts)
    n_reached = sum(1 for r in rows if r["reachable"] == "yes")
    fh.write(f"- Total scripts in `scripts/`: {n_total}\n")
    fh.write(f"- Reachable from entry points: {n_reached}\n")
    fh.write(f"- Not reachable: {n_total - n_reached}\n")
    fh.write("\n")
    fh.write("## Unreachable scripts (samples)\n")
    for r in rows:
        if r["reachable"] == "no":
            fh.write(f"- {r['script']}\n")
    fh.write("\n")
    fh.write("## Notes\n")
    fh.write(
        "- This is a static, best-effort analysis based on lexical matches in scripts, tests, docs and README.\n"
    )
    fh.write(
        "- Some paths depend on runtime flags, environment wrappers, or optional imports and might be missed.\n"
    )

print(f"Wrote CSV: {csv_out}")
print(f"Wrote report: {report}")
print("Done.")
