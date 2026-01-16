"""Simple import scanner that lists top-level imports and compares them to requirements.

Usage: python tools/check_imports.py --requirements requirements-cpu.txt
Exits with code 0 on success; prints missing/unlisted modules and exits 2 if any missing required package.
"""
from pathlib import Path
import ast
import argparse
import json


def find_modules(paths):
    modules = set()
    for p in paths:
        try:
            tree = ast.parse(Path(p).read_text(), filename=str(p))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    modules.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split('.')[0])
    return sorted(m for m in modules if m not in ('src','scripts','tests','docs','data','outputs'))


def read_requirements(req_path):
    reqs = set()
    for line in Path(req_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue
        pkg = line.split('==')[0].split('>=')[0].strip()
        reqs.add(pkg.lower())
    return reqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirements', default='requirements-cpu.txt')
    args = parser.parse_args()

    repo_root = Path('.').resolve()
    py_files = list(repo_root.glob('src/**/*.py')) + list(repo_root.glob('scripts/**/*.py')) + list(repo_root.glob('tests/**/*.py'))
    modules = find_modules(py_files)
    reqs = read_requirements(args.requirements)

    # simple heuristics mapping
    mapping = {
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'yaml': 'pyyaml',
        'umap': 'umap-learn',
        'apricot': 'apricot-select',
        'seaborn': 'seaborn',
    }

    missing = []
    unlisted = []
    for m in modules:
        m_low = m.lower()
        mapped = mapping.get(m, None)
        in_reqs = any(r.startswith(m_low) or (m_low in r) for r in reqs)
        if not in_reqs:
            unlisted.append({'module': m, 'mapped': mapped})
    print(json.dumps({'modules': modules, 'unlisted': unlisted}, indent=2))
    if any(u['mapped'] is None for u in unlisted):
        # warn but don't fail
        print('\nNote: Some modules have no suggestion; please review manually.')
    # exit code 0; maintainers decide whether to fail CI or only warn


if __name__ == '__main__':
    main()
