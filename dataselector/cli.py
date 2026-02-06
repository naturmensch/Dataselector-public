"""
Dataselector CLI - Pattern C decorator-based dispatcher.

All commands are registered via @cli_command decorators in their respective modules.
This file only imports modules (triggering registration) and dispatches to the right handler.
"""

from __future__ import annotations

# Import data module
import dataselector.data.build_tiles
import dataselector.tools.archive
import dataselector.tools.audit

# Import ALL tools modules to trigger @cli_command decorator execution
import dataselector.tools.check
import dataselector.tools.clean
import dataselector.tools.docs_link
import dataselector.workflows.adaptive_pipeline

# Import ALL workflow modules to trigger @cli_command decorator execution
import dataselector.workflows.autoscale
import dataselector.workflows.benchmark_sampling
import dataselector.workflows.bootstrap
import dataselector.workflows.compare_samplers
import dataselector.workflows.final_selection
import dataselector.workflows.fine_sweep
import dataselector.workflows.generate_reports
import dataselector.workflows.optuna_autoscale
import dataselector.workflows.optuna_optimize
import dataselector.workflows.sampler_suite
import dataselector.workflows.thesis_pipeline
import dataselector.workflows.thesis_sampler_suite
import dataselector.workflows.tune_weights
import dataselector.workflows.validation
import dataselector.workflows.xxl

# Import decorator infrastructure
from dataselector.cli_decorators import (
    build_parser_from_decorators,
    dispatch_from_decorators,
)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point using decorator-based dispatcher.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    # Build parser from all registered @cli_command decorators
    parser = build_parser_from_decorators()

    # Parse arguments
    args = parser.parse_args(argv)

    # Dispatch to registered command handler
    return dispatch_from_decorators(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
