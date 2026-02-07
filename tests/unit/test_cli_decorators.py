"""
Unit tests for CLI decorator system.
"""

import argparse

import pytest

from dataselector.cli_decorators import (
    _CLI_COMMANDS,
    ArgDef,
    build_parser_from_decorators,
    cli_command,
    dispatch_from_decorators,
    get_registered_commands,
)


@pytest.fixture(autouse=True)
def clear_commands():
    """Isolate command registry for decorator tests and restore global state."""
    original_commands = _CLI_COMMANDS.copy()
    _CLI_COMMANDS.clear()
    yield
    _CLI_COMMANDS.clear()
    _CLI_COMMANDS.update(original_commands)


class TestArgDef:
    """Test ArgDef dataclass."""

    def test_argdef_creation(self):
        """Test creating ArgDef."""
        arg = ArgDef(
            name="csv",
            type=str,
            help="CSV path",
            required=True,
        )
        assert arg.name == "csv"
        assert arg.type is str
        assert arg.help == "CSV path"
        assert arg.required is True
        assert arg.default is None

    def test_argdef_with_defaults(self):
        """Test ArgDef with all optional fields."""
        arg = ArgDef(
            name="n_trials",
            type=int,
            help="Number of trials",
            default=100,
            nargs="+",
            choices=[10, 20, 50, 100],
        )
        assert arg.default == 100
        assert arg.nargs == "+"
        assert arg.choices == [10, 20, 50, 100]


class TestCliCommandDecorator:
    """Test @cli_command decorator."""

    def test_simple_command(self):
        """Test registering a simple command."""

        @cli_command("test", help="Test command", args={"csv": {"type": str}})
        def main(csv: str) -> int:
            return 0

        assert "test" in _CLI_COMMANDS
        cmd_def = _CLI_COMMANDS["test"]
        assert cmd_def.name == "test"
        assert cmd_def.help == "Test command"
        assert "csv" in cmd_def.args

    def test_command_with_multiple_args(self):
        """Test command with multiple arguments."""

        @cli_command(
            "autoscale",
            help="Run autoscale",
            args={
                "csv": {"type": str, "help": "CSV path"},
                "n_trials": {"type": int, "help": "Trials", "nargs": "+"},
                "smoke": {"type": bool, "action": "store_true"},
            },
        )
        def main(csv: str, n_trials: list[int], smoke: bool = False) -> int:
            return 0

        cmd_def = _CLI_COMMANDS["autoscale"]
        assert len(cmd_def.args) == 3
        assert cmd_def.args["csv"].type is str
        assert cmd_def.args["n_trials"].type is int
        assert cmd_def.args["n_trials"].nargs == "+"
        assert cmd_def.args["smoke"].action == "store_true"

    def test_command_with_default_values(self):
        """Test command with default parameter values."""

        @cli_command(
            "test",
            args={
                "n_samples": {"type": int, "default": 34},
            },
        )
        def main(n_samples: int = 34) -> int:
            return 0

        cmd_def = _CLI_COMMANDS["test"]
        assert cmd_def.args["n_samples"].default == 34

    def test_missing_function_param_raises_error(self):
        """Test that declaring arg not in function signature raises error."""
        with pytest.raises(ValueError, match="not in function signature"):

            @cli_command(
                "bad",
                args={"csv": {"type": str}},
            )
            def main() -> int:  # Missing csv parameter
                return 0

    def test_undocumented_required_param_raises_error(self):
        """Test that function param without default and without arg def raises error."""
        with pytest.raises(ValueError, match="not declared"):

            @cli_command("bad", args={})
            def main(csv: str) -> int:  # csv not documented and no default
                return 0

    def test_undocumented_optional_param_allowed(self):
        """Test that function param with default doesn't need arg declaration."""

        @cli_command("good", args={})
        def main(csv: str = "default.csv") -> int:  # Has default, so OK
            return 0

        # Should not raise
        assert "good" in _CLI_COMMANDS

    def test_decorator_returns_original_function(self):
        """Test that decorator returns original function."""

        @cli_command("test", args={})
        def main() -> int:
            return 42

        # Can still call original function
        assert main() == 42


class TestParserGeneration:
    """Test build_parser_from_decorators."""

    def test_parser_raises_if_no_commands(self):
        """Test that parser generation fails if no commands registered."""
        with pytest.raises(RuntimeError, match="No CLI commands registered"):
            build_parser_from_decorators()

    def test_parser_with_single_command(self):
        """Test parser generation with one command."""

        @cli_command("test", help="Test", args={"csv": {"type": str}})
        def main(csv: str) -> int:
            return 0

        parser = build_parser_from_decorators()
        ns = parser.parse_args(["test", "--csv", "data.csv"])
        assert ns.cmd == "test"
        assert ns.csv == "data.csv"

    def test_parser_with_multiple_commands(self):
        """Test parser generation with multiple commands."""

        @cli_command("cmd1", args={"x": {"type": str}})
        def main1(x: str) -> int:
            return 0

        @cli_command("cmd2", args={"y": {"type": int}})
        def main2(y: int) -> int:
            return 0

        parser = build_parser_from_decorators()
        ns1 = parser.parse_args(["cmd1", "--x", "hello"])
        assert ns1.cmd == "cmd1"
        assert ns1.x == "hello"

        ns2 = parser.parse_args(["cmd2", "--y", "42"])
        assert ns2.cmd == "cmd2"
        assert ns2.y == 42

    def test_parser_converts_underscores_to_hyphens(self):
        """Test that parameter names are converted from snake_case to hyphenated."""

        @cli_command("test", args={"n_trials": {"type": int}})
        def main(n_trials: int) -> int:
            return 0

        parser = build_parser_from_decorators()
        ns = parser.parse_args(["test", "--n-trials", "100"])
        assert ns.n_trials == 100

    def test_parser_handles_bool_flags(self):
        """Test that bool arguments become store_true flags."""

        @cli_command("test", args={"smoke": {"type": bool, "action": "store_true"}})
        def main(smoke: bool = False) -> int:
            return 0

        parser = build_parser_from_decorators()

        # Without flag
        ns1 = parser.parse_args(["test"])
        assert ns1.smoke is False

        # With flag
        ns2 = parser.parse_args(["test", "--smoke"])
        assert ns2.smoke is True

    def test_parser_handles_nargs(self):
        """Test that nargs is passed to argparse."""

        @cli_command("test", args={"values": {"type": int, "nargs": "+"}})
        def main(values: list[int]) -> int:
            return 0

        parser = build_parser_from_decorators()
        ns = parser.parse_args(["test", "--values", "1", "2", "3"])
        assert ns.values == [1, 2, 3]

    def test_parser_handles_choices(self):
        """Test that choices are validated."""

        @cli_command(
            "test", args={"sampler": {"type": str, "choices": ["tpe", "cmaes"]}}
        )
        def main(sampler: str) -> int:
            return 0

        parser = build_parser_from_decorators()

        # Valid choice
        ns = parser.parse_args(["test", "--sampler", "tpe"])
        assert ns.sampler == "tpe"

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["test", "--sampler", "invalid"])


class TestDispatcher:
    """Test dispatch_from_decorators."""

    def test_dispatch_simple_command(self):
        """Test dispatching a simple command."""

        @cli_command("test", args={"csv": {"type": str}})
        def main(csv: str) -> int:
            return 42 if csv == "test.csv" else 1

        parser = build_parser_from_decorators()
        ns = parser.parse_args(["test", "--csv", "test.csv"])
        result = dispatch_from_decorators(ns)
        assert result == 42

    def test_dispatch_with_multiple_args(self):
        """Test dispatch with multiple arguments."""

        @cli_command(
            "calc",
            args={
                "x": {"type": int},
                "y": {"type": int},
            },
        )
        def main(x: int, y: int) -> int:
            return x + y

        parser = build_parser_from_decorators()
        ns = parser.parse_args(["calc", "--x", "10", "--y", "32"])
        result = dispatch_from_decorators(ns)
        assert result == 42

    def test_dispatch_with_defaults(self):
        """Test dispatch with default values."""

        @cli_command(
            "test",
            args={
                "n_samples": {"type": int, "default": 34},
                "smoke": {"type": bool, "action": "store_true"},
            },
        )
        def main(n_samples: int = 34, smoke: bool = False) -> int:
            return n_samples if not smoke else 0

        parser = build_parser_from_decorators()

        # With defaults
        ns1 = parser.parse_args(["test"])
        assert dispatch_from_decorators(ns1) == 34

        # Override default
        ns2 = parser.parse_args(["test", "--n-samples", "100"])
        assert dispatch_from_decorators(ns2) == 100

        # With flag
        ns3 = parser.parse_args(["test", "--smoke"])
        assert dispatch_from_decorators(ns3) == 0

    def test_dispatch_raises_if_no_function(self):
        """Test that dispatch raises error if command not found."""
        ns = argparse.Namespace(cmd="missing", _cmd_func=None)
        with pytest.raises(ValueError, match="No function"):
            dispatch_from_decorators(ns)


class TestGetRegisteredCommands:
    """Test get_registered_commands function."""

    def test_get_all_commands(self):
        """Test retrieving all registered commands."""

        @cli_command("cmd1", args={})
        def main1() -> int:
            return 0

        @cli_command("cmd2", args={})
        def main2() -> int:
            return 0

        commands = get_registered_commands()
        assert len(commands) == 2
        assert "cmd1" in commands
        assert "cmd2" in commands

    def test_get_commands_returns_copy(self):
        """Test that returned dict is a copy (modifications don't affect registry)."""

        @cli_command("test", args={})
        def main() -> int:
            return 0

        commands1 = get_registered_commands()
        commands1["test"].help = "Modified"

        # Original registry unchanged
        commands2 = get_registered_commands()
        assert commands2["test"].help == ""


class TestIntegration:
    """Integration tests for full CLI workflow."""

    def test_full_workflow(self):
        """Test complete workflow: register → parse → dispatch."""

        @cli_command(
            "sum",
            help="Sum two numbers",
            args={
                "a": {"type": int, "help": "First number"},
                "b": {"type": int, "help": "Second number"},
            },
        )
        def main(a: int, b: int) -> int:
            return a + b

        # Build parser
        parser = build_parser_from_decorators()

        # Parse args
        ns = parser.parse_args(["sum", "--a", "10", "--b", "32"])

        # Dispatch
        result = dispatch_from_decorators(ns)
        assert result == 42

    def test_help_text_available(self):
        """Test that help text is available."""

        @cli_command(
            "test",
            help="Test command",
            args={"csv": {"type": str, "help": "CSV file"}},
        )
        def main(csv: str) -> int:
            return 0

        parser = build_parser_from_decorators()

        # Help should work without error
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
