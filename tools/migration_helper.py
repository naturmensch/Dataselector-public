#!/usr/bin/env python3
"""
Migration Helper: Convert argparse CLI to @cli_command decorator pattern.

Usage:
    python tools/migrate_cli_command.py <file.py> <command_name>

Example:
    python tools/migrate_cli_command.py dataselector/workflows/adaptive_pipeline.py adaptive-pipeline

This script:
1. Parses the old main() function with argparse.ArgumentParser
2. Extracts all add_argument() calls
3. Generates @cli_command decorator with all args
4. Generates new main() function signature
5. Outputs migration report
"""

import ast
import re
import sys
from pathlib import Path


def extract_argparse_calls(file_path: str) -> list[dict]:
    """
    Extract all parser.add_argument() calls from a file.

    Returns list of dicts with: name, type, default, nargs, action, help
    """
    with open(file_path) as f:
        content = f.read()

    # Find all parser.add_argument() calls
    pattern = r"parser\.add_argument\((.*?)\)"
    matches = re.finditer(pattern, content, re.DOTALL)

    args_list = []
    for match in matches:
        args_str = match.group(1)

        # Parse first arg (positional or --name)
        first_arg_match = re.search(r'["\']([^"\']+)["\']', args_str)
        if not first_arg_match:
            continue
        arg_name = first_arg_match.group(1)

        # Clean up arg name (remove -- prefix)
        if arg_name.startswith("--"):
            arg_name = arg_name[2:]
        arg_name_py = arg_name.replace("-", "_")  # Convert to Python identifier

        # Extract metadata
        arg_info = {
            "cli_name": arg_name,
            "py_name": arg_name_py,
            "type": None,
            "default": None,
            "nargs": None,
            "action": None,
            "help": None,
        }

        # Extract type
        type_match = re.search(r"type=(\w+)", args_str)
        if type_match:
            type_name = type_match.group(1)
            arg_info["type"] = (
                "int"
                if type_name == "int"
                else "str" if type_name == "str" else type_name
            )

        # Extract default
        default_match = re.search(r"default=([^,\)]+)", args_str)
        if default_match:
            default_val = default_match.group(1).strip()
            arg_info["default"] = default_val

        # Extract nargs
        nargs_match = re.search(r'nargs=["\']?([^,\)]+)["\']?', args_str)
        if nargs_match:
            arg_info["nargs"] = nargs_match.group(1)

        # Extract action
        action_match = re.search(r'action=["\']?([^,\)]+)["\']?', args_str)
        if action_match:
            arg_info["action"] = action_match.group(1)

        # Extract help
        help_match = re.search(r'help=["\']([^"\']+)["\']', args_str)
        if help_match:
            arg_info["help"] = help_match.group(1)

        args_list.append(arg_info)

    return args_list


def generate_decorator(
    command_name: str, args_list: list[dict], help_text: str = ""
) -> str:
    """Generate @cli_command decorator string."""
    decorator_lines = [
        f"@cli_command(",
        f'    "{command_name}",',
        f'    help="{help_text or "TODO: Add help text"}",',
        f"    args={{",
    ]

    for arg in args_list:
        decorator_lines.append(f'        "{arg["py_name"]}": {{')
        if arg["type"]:
            decorator_lines.append(f'            "type": {arg["type"]},')
        if arg["default"]:
            decorator_lines.append(f'            "default": {arg["default"]},')
        if arg["nargs"]:
            decorator_lines.append(f'            "nargs": {arg["nargs"]},')
        if arg["action"]:
            decorator_lines.append(f'            "action": "{arg["action"]}",')
        if arg["help"]:
            decorator_lines.append(
                f'            "help": "{arg["help"].replace('"', '\\"')}",'
            )
        decorator_lines.append(f"        }},")

    decorator_lines.append(f"    }},")
    decorator_lines.append(f")")

    return "\n".join(decorator_lines)


def generate_function_signature(args_list: list[dict]) -> str:
    """Generate new main() function signature."""
    params = []
    for arg in args_list:
        default_val = arg["default"] if arg["default"] else "None"

        # Determine type hint
        type_hint = "str"
        if arg["type"] == "int":
            type_hint = "int"
        elif arg["nargs"]:
            type_hint = "list[str]" if arg["type"] == "str" else "list[int]"

        if arg["action"] == "store_true":
            type_hint = "bool"
            default_val = "False"

        params.append(
            f'    {arg["py_name"]}: {type_hint} | None = {default_val}'
            if arg["default"] is None
            else f'    {arg["py_name"]}: {type_hint} = {default_val}'
        )

    sig = "def main(\n"
    sig += ",\n".join(params)
    sig += ",\n) -> int:"

    return sig


def generate_migration_report(file_path: str, args_list: list[dict]) -> str:
    """Generate human-readable migration report."""
    report = f"""
=== Migration Report ===
File: {file_path}
Arguments found: {len(args_list)}

Arguments to migrate:
"""
    for i, arg in enumerate(args_list, 1):
        report += f"\n{i}. --{arg['cli_name']} ({arg['py_name']})"
        if arg["type"]:
            report += f" [type: {arg['type']}]"
        if arg["default"]:
            report += f" [default: {arg['default']}]"
        if arg["nargs"]:
            report += f" [nargs: {arg['nargs']}]"
        if arg["action"]:
            report += f" [action: {arg['action']}]"

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python migration_helper.py <file.py> <command_name> [help_text]")
        print(
            "Example: python migration_helper.py dataselector/workflows/adaptive_pipeline.py adaptive-pipeline"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    command_name = (
        sys.argv[2] if len(sys.argv) > 2 else Path(file_path).stem.replace("_", "-")
    )
    help_text = sys.argv[3] if len(sys.argv) > 3 else ""

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Extract arguments
    args_list = extract_argparse_calls(file_path)

    # Generate outputs
    print(generate_migration_report(file_path, args_list))
    print("\n=== Generated Decorator ===")
    print(generate_decorator(command_name, args_list, help_text))
    print("\n=== Generated Function Signature ===")
    print(generate_function_signature(args_list))


if __name__ == "__main__":
    main()
