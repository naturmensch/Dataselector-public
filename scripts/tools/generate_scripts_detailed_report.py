#!/usr/bin/env python3
"""Generate a detailed per-Python-file report:
- top-level imports (module names)
- top-level function and class names
- whether it contains a __main__ guard
- subprocess-like calls (subprocess.*, os.system, Popen etc.)

Outputs Markdown to docs/all_scripts_overview_detailed.md
"""

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {"tests", "archive", "docs/.*/archive", ".git", "node_modules", "archive_local"}

out_lines = ["# Detailed Python Scripts Report\n"]

py_files = sorted(ROOT.glob("**/*.py"))

def is_excluded(p: Path):
    parts = set(p.parts)
    if "tests" in p.parts or any(str(p).startswith(str(ROOT / d)) for d in ["archive", "archive_local"]):
        return True
    # exclude tests and test helpers
    if "/tests/" in str(p) or str(p).endswith("_test.py"):
        return True
    return False

for p in py_files:
    rel = p.relative_to(ROOT)
    if is_excluded(p):
        continue
    try:
        src = p.read_text(encoding="utf8")
    except Exception as e:
        out_lines.append(f"## {rel} - Could not read: {e}\n")
        continue
    try:
        tree = ast.parse(src)
    except Exception as e:
        out_lines.append(f"## {rel} - Could not parse AST: {e}\n")
        continue

    imports = set()
    imported_names = set()
    top_level_funcs = []
    top_level_classes = []
    has_main_guard = False
    subprocess_calls = set()
    click_or_argparse = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
                imported_names.add(n.asname or n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                imports.add(mod.split(".")[0])
            for n in node.names:
                imported_names.add(n.asname or n.name)
        elif isinstance(node, ast.FunctionDef):
            top_level_funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            top_level_classes.append(node.name)
        elif isinstance(node, ast.If):
            # detect __main__ guard
            # naive check
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = ""
            if "__name__" in cond and "__main__" in cond:
                has_main_guard = True

    # traverse all calls for subprocess-like usage
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # function could be ast.Attribute or ast.Name
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "subprocess":
                        subprocess_calls.add(f"subprocess.{node.func.attr}")
                # e.g., sp.run when sp is alias
                elif isinstance(node.func.value, ast.Name):
                    pass
            elif isinstance(node.func, ast.Name):
                if node.func.id in {"Popen", "run", "call", "check_output", "check_call"}:
                    subprocess_calls.add(node.func.id)
                if node.func.id == "system":
                    subprocess_calls.add("os.system")
            self.generic_visit(node)

    CallVisitor().visit(tree)

    # heuristics: look for os.system
    if "os.system" in src:
        subprocess_calls.add("os.system")

    if "subprocess" in imports or "subprocess" in src:
        # leave it as-is; already captured
        pass

    if "click" in imports or "argparse" in imports:
        if "click" in imports:
            click_or_argparse.add("click")
        if "argparse" in imports:
            click_or_argparse.add("argparse")

    # build markdown section
    out_lines.append(f"## {rel}\n")
    out_lines.append(f"- **Path:** `{rel}`\n")
    out_lines.append(f"- **Imports (top-level modules):** {', '.join(sorted(imports)) if imports else '*(none detected)*'}\n")
    out_lines.append(f"- **Top-level functions:** {', '.join(top_level_funcs) if top_level_funcs else '*(none)*'}\n")
    out_lines.append(f"- **Top-level classes:** {', '.join(top_level_classes) if top_level_classes else '*(none)*'}\n")
    out_lines.append(f"- **Has __main__ guard:** {'Yes' if has_main_guard else 'No'}\n")
    out_lines.append(f"- **Subprocess-like calls detected:** {', '.join(sorted(subprocess_calls)) if subprocess_calls else '*(none)*'}\n")
    if click_or_argparse:
        out_lines.append(f"- **CLI libs detected:** {', '.join(click_or_argparse)}\n")
    out_lines.append("\n")

# write output
out_path = ROOT / "docs" / "all_scripts_overview_detailed.md"
out_path.write_text("\n".join(out_lines), encoding="utf8")
print("WROTE", out_path)
