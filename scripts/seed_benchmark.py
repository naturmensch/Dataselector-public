#!/usr/bin/env python3
"""Wrapper script for seed/runtime benchmark.

Delegates to the maintained workflow implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic seeded mode against non-deterministic mode."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of seeds (default workflow seeds are used when omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to outputs/).",
    )
    parser.add_argument(
        "--subset-n",
        type=int,
        default=200,
        help="Feature subset size used for timing benchmark.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dataselector.workflows.compare_samplers import benchmark_seed

    out_csv = benchmark_seed(
        seeds=args.seeds,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        subset_n=int(args.subset_n),
    )
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
