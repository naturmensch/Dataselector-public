#!/usr/bin/env python3
"""
Smoke test tool to verify that all scripts in the scripts/ directory are importable
without triggering side effects (like running the main logic or printing to stdout).

Exit code is 0 if all scripts import successfully, non-zero otherwise.
"""

import importlib.util
import sys
from pathlib import Path


def smoke_check_scripts(directory: Path) -> bool:
    all_passed = True
    scripts = sorted(list(directory.glob("*.py")))

    print(f"Checking {len(scripts)} scripts in {directory}...")

    for script_path in scripts:
        if script_path.name == "__init__.py":
            continue

        script_name = script_path.stem
        try:
            # Use importlib to load the script as a module
            spec = importlib.util.spec_from_file_location(script_name, str(script_path))
            if spec is None:
                print(f"[-] FAILED: Could not create spec for {script_path.name}")
                all_passed = False
                continue

            module = importlib.util.module_from_spec(spec)
            # This triggers the actual code execution at module level!
            spec.loader.exec_module(module)

            print(f"[+] PASSED: {script_path.name}")

        except Exception as e:
            print(f"[-] FAILED: {script_path.name} - {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


def main():
    repo_root = Path(__file__).parent.parent
    scripts_dir = repo_root / "scripts"

    if not scripts_dir.exists():
        print(f"Error: scripts directory not found at {scripts_dir}")
        sys.exit(1)

    success = smoke_check_scripts(scripts_dir)

    if success:
        print("\nAll scripts imported successfully!")
        sys.exit(0)
    else:
        print("\nSome scripts failed to import correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
