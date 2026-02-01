from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataselector")
    sub = p.add_subparsers(dest="cmd", required=True)

    autoscale = sub.add_parser("autoscale", help="Run Optuna autoscale phase")
    autoscale.add_argument("args", nargs=argparse.REMAINDER)

    suite = sub.add_parser(
        "sampler-suite", help="Run the sampler suite (optionally includes autoscale)"
    )
    suite.add_argument("args", nargs=argparse.REMAINDER)

    xxl = sub.add_parser("xxl", help="Run XXL thesis pipeline")
    xxl.add_argument("args", nargs=argparse.REMAINDER)

    xxl_mon = sub.add_parser(
        "xxl-monitor",
        help="Run XXL full-run monitor (scripts/xxl_full_run_monitor.py)",
    )
    xxl_mon.add_argument("args", nargs=argparse.REMAINDER)

    opt = sub.add_parser(
        "optuna-optimize",
        help="Run Optuna optimization runner (scripts/optuna_optimize.py)",
    )
    opt.add_argument("args", nargs=argparse.REMAINDER)

    fin = sub.add_parser(
        "final-selection",
        help="Run final selection runner (scripts/final_selection.py)",
    )
    fin.add_argument("args", nargs=argparse.REMAINDER)

    pipe = sub.add_parser(
        "thesis-pipeline",
        help=(
            "Run full thesis pipeline orchestrator "
            "(scripts/run_complete_thesis_pipeline.sh)"
        ),
    )
    pipe.add_argument("args", nargs=argparse.REMAINDER)

    cs = sub.add_parser(
        "compare-samplers",
        help="Compare samplers across seeds (scripts/compare_samplers_multi_seed.py)",
    )
    cs.add_argument("args", nargs=argparse.REMAINDER)

    bench = sub.add_parser(
        "benchmark-sampling",
        help=(
            "Benchmark initial sampling methods and write an exploration plan artifact"
        ),
    )
    bench.add_argument("args", nargs=argparse.REMAINDER)

    rep = sub.add_parser(
        "generate-reports",
        help="Generate summary plots/reports (scripts/generate_reports.py)",
    )
    rep.add_argument("args", nargs=argparse.REMAINDER)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    ns = p.parse_args(argv)

    if ns.cmd == "autoscale":
        from dataselector.workflows.autoscale import main as run

        # argparse can't reliably pass through "--" with REMAINDER in all cases.
        # We strip a leading "--" if present.
        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "sampler-suite":
        from dataselector.workflows.sampler_suite import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "xxl":
        from dataselector.workflows.xxl import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "xxl-monitor":
        from dataselector.workflows.xxl_monitor import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "optuna-optimize":
        from dataselector.workflows.optuna_optimize import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "final-selection":
        from dataselector.workflows.final_selection import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "thesis-pipeline":
        from dataselector.workflows.thesis_pipeline import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "compare-samplers":
        from dataselector.workflows.compare_samplers import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "benchmark-sampling":
        from dataselector.workflows.benchmark_sampling import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "generate-reports":
        from dataselector.workflows.generate_reports import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    raise SystemExit(f"Unknown command: {ns.cmd}")
