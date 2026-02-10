#!/usr/bin/env python3
"""Wrapper script for seeded-vs-unseeded selection comparison.

This script intentionally delegates to the maintained workflow implementation
instead of reintroducing legacy script-era logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs Hamburg-seeded selection on canonical metadata."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional pipeline config path (defaults to config/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to outputs/seed_benchmark).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dataselector.workflows.compare_samplers import compare_seeded_vs_unseeded

    config_path = Path(args.config_path) if args.config_path else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    out = compare_seeded_vs_unseeded(config_path=config_path, output_dir=output_dir)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
