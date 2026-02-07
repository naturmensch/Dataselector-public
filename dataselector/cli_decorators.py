"""
CLI Command Decorator System

Provides @cli_command decorator for registering workflows as CLI commands.
Automatically generates argparse infrastructure from command metadata.

Example:
    @cli_command("autoscale", help="Run autoscale workflow", args={
        "csv": {"type": str, "help": "CSV path", "default": None},
        "n_trials": {"type": int, "help": "Trials per stage", "nargs": "+"},
        "smoke": {"type": bool, "action": "store_true"},
    })
    def main(csv: str | None = None, n_trials: list[int] = None, smoke: bool = False) -> int:
        ...

    # Auto-generated CLI:
    # dataselector autoscale --csv data.csv --n-trials 20 40 80 --smoke
"""

from __future__ import annotations

import argparse
import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

# Type for command functions
T = TypeVar("T", bound=Callable[..., int])

# Global registry of CLI commands
_CLI_COMMANDS: dict[str, CommandDef] = {}


@dataclass
class ArgDef:
    """Definition of a single CLI argument."""

    name: str
    """Argument name (python identifier, converted to --hyphenated)"""

    type: type
    """Python type (str, int, float, bool)"""

    help: str
    """Help text displayed in --help"""

    required: bool = False
    """Whether argument is required"""

    default: Any = None
    """Default value if not provided"""

    nargs: str | int | None = None
    """Argument multiplicity (None, "+", "*", "?", or N)"""

    choices: list[Any] | None = None
    """Allowed choices for the argument"""

    action: str | None = None
    """Action for argparse (e.g., "store_true" for flags)"""


@dataclass
class CommandDef:
    """Definition of a registered CLI command."""

    name: str
    """Command name (how user invokes it)"""

    help: str
    """Help text for --help"""

    args: dict[str, ArgDef]
    """Registered arguments for this command"""

    func: Callable[..., int]
    """Function to execute when command is called"""


def cli_command(
    name: str,
    help: str = "",
    args: dict[str, dict[str, Any]] | None = None,
) -> Callable[[T], T]:
    """
    Decorator to register a workflow function as a CLI command.

    Args:
        name: CLI command name (e.g., "autoscale")
        help: Help text for this command
        args: Dict mapping argument names to argument specifications.
              Each spec can contain:
              - "type": Python type (str, int, float, bool) [default: str]
              - "help": Help text
              - "required": Whether required [default: False]
              - "default": Default value
              - "nargs": Multiplicity ("+", "*", "?", or int) [default: None]
              - "choices": List of allowed values [default: None]
              - "action": argparse action ("store_true", etc) [default: None]

    Returns:
        Decorator function that registers the command and returns original function.

    Raises:
        ValueError: If function signature doesn't match declared arguments.

    Example:
        @cli_command("autoscale", help="Run autoscale", args={
            "csv": {"type": str, "help": "CSV path"},
            "n_trials": {"type": int, "nargs": "+", "help": "Trials"},
            "smoke": {"action": "store_true", "help": "Smoke mode"},
        })
        def main(csv: str, n_trials: list[int], smoke: bool = False) -> int:
            ...
    """

    def decorator(func: Callable[..., int]) -> Callable[..., int]:
        # Parse function signature
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())

        # Parse argument specifications
        parsed_args: dict[str, ArgDef] = {}
        for arg_name, arg_spec in (args or {}).items():
            if arg_name not in func_params:
                raise ValueError(
                    f"Argument '{arg_name}' declared but not in function signature"
                )

            parsed_args[arg_name] = ArgDef(
                name=arg_name,
                type=arg_spec.get("type", str),
                help=arg_spec.get("help", ""),
                required=arg_spec.get("required", False),
                default=arg_spec.get("default", None),
                nargs=arg_spec.get("nargs", None),
                choices=arg_spec.get("choices", None),
                action=arg_spec.get("action", None),
            )

        # Verify all function parameters are documented or have defaults
        for param_name, param in sig.parameters.items():
            if param_name not in parsed_args:
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{param_name}' in function signature "
                        f"but not declared in @cli_command args"
                    )

        # Register command globally
        cmd_def = CommandDef(
            name=name,
            help=help,
            args=parsed_args,
            func=func,
        )
        _CLI_COMMANDS[name] = cmd_def

        # Return original function unchanged
        return func

    return decorator


def build_parser_from_decorators() -> argparse.ArgumentParser:
    """
    Generate argparse ArgumentParser from all registered @cli_command decorators.

    Returns:
        ArgumentParser with subparsers for each registered command.
    """
    parser = argparse.ArgumentParser(
        prog="dataselector",
        description="Data selection toolkit for KDR100",
    )

    if not _CLI_COMMANDS:
        raise RuntimeError(
            "No CLI commands registered. "
            "Import command modules to register @cli_command decorators."
        )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for cmd_name in sorted(_CLI_COMMANDS.keys()):
        cmd_def = _CLI_COMMANDS[cmd_name]

        # Create subparser for this command
        subparser = subparsers.add_parser(cmd_name, help=cmd_def.help)

        # Add arguments to subparser
        for arg_name, arg_def in cmd_def.args.items():
            # Convert snake_case to --hyphenated-case
            cli_arg_name = f"--{arg_def.name.replace('_', '-')}"

            # Build kwargs for add_argument()
            kwargs: dict[str, Any] = {
                "help": arg_def.help,
            }

            # Add default only if not None (avoid conflicts with action)
            if arg_def.default is not None:
                kwargs["default"] = arg_def.default

            # Add required if specified
            if arg_def.required:
                kwargs["required"] = True

            # Handle boolean flags specially
            if arg_def.type is bool:
                if arg_def.action == "store_true":
                    kwargs["action"] = "store_true"
                elif arg_def.action == "store_false":
                    kwargs["action"] = "store_false"
                else:
                    # Default: treat as flag
                    kwargs["action"] = "store_true"
                # Do NOT add 'type' for boolean flags (incompatible with store_true/false)
            else:
                kwargs["type"] = arg_def.type
                # Add action for non-bool types if specified
                if arg_def.action is not None:
                    kwargs["action"] = arg_def.action

            # Add optional arguments
            if arg_def.nargs is not None:
                kwargs["nargs"] = arg_def.nargs

            if arg_def.choices is not None:
                kwargs["choices"] = arg_def.choices

            # Remove None/empty values (already handled above, but double-check)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Add argument to subparser
            subparser.add_argument(cli_arg_name, **kwargs)

        # Store reference to command function for later dispatch
        subparser.set_defaults(_cmd_func=cmd_def.func)

    return parser


def dispatch_from_decorators(ns: argparse.Namespace) -> int:
    """
    Execute the appropriate command function based on parsed arguments.

    Args:
        ns: Namespace from parser.parse_args()

    Returns:
        Exit code from command function

    Raises:
        ValueError: If no command function found for namespace
    """
    cmd_func = getattr(ns, "_cmd_func", None)
    if cmd_func is None:
        raise ValueError(f"No function registered for command: {ns.cmd}")

    # Extract command arguments from namespace
    # Remove internal metadata (cmd, _cmd_func)
    cmd_dict = {
        k: v for k, v in vars(ns).items() if not k.startswith("_") and k != "cmd"
    }

    # Convert hyphenated names back to underscores for function parameters
    cmd_dict = {k.replace("-", "_"): v for k, v in cmd_dict.items()}

    # Get function signature to filter kwargs
    sig = inspect.signature(cmd_func)
    func_params = set(sig.parameters.keys())

    # Pass only arguments that function accepts
    filtered_kwargs = {k: v for k, v in cmd_dict.items() if k in func_params}

    # Execute command function
    return cmd_func(**filtered_kwargs)


def get_registered_commands() -> dict[str, CommandDef]:
    """
    Get all registered commands.

    Returns:
        Deep copy of dict mapping command names to CommandDef objects.
    """
    return copy.deepcopy(_CLI_COMMANDS)
