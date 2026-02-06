from __future__ import annotations

from dataselector.cli_decorators import cli_command


@cli_command(
    "sampler-suite",
    help="Thesis-grade sampler evaluation suite (alias for thesis-sampler-suite)",
    args={},
)
def main() -> int:
    """Alias command that delegates to thesis-sampler-suite.

    This is kept for backwards compatibility.
    Use 'dataselector thesis-sampler-suite' directly instead.
    """
    from dataselector.workflows import thesis_sampler_suite

    print("Note: 'sampler-suite' is an alias. Use 'thesis-sampler-suite' directly.")
    return thesis_sampler_suite.main()
