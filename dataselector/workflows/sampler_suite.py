from __future__ import annotations

from dataselector.cli_decorators import cli_command
from dataselector.workflows import thesis_sampler_suite


@cli_command(
    "sampler-suite",
    help="Thesis-grade sampler evaluation suite (alias for thesis-sampler-suite)",
    args=thesis_sampler_suite.THESIS_SAMPLER_SUITE_ARGS,
)
def main(
    seeds: list[int] | None = None,
    n_trials: int = 1000,
    datasets: list[str] | None = None,
    samplers: list[str] | None = None,
    sequential: bool = False,
    n_trials_full: int = 2000,
    n_candidates: int | None = None,
    autoscale: bool = True,
) -> int:
    """Alias command that delegates to thesis-sampler-suite.

    This is kept for backwards compatibility.
    Use 'dataselector thesis-sampler-suite' directly instead.
    """
    print("Note: 'sampler-suite' is an alias. Use 'thesis-sampler-suite' directly.")
    return thesis_sampler_suite.main(
        seeds=seeds,
        n_trials=n_trials,
        datasets=datasets,
        samplers=samplers,
        sequential=sequential,
        n_trials_full=n_trials_full,
        n_candidates=n_candidates,
        autoscale=autoscale,
    )
