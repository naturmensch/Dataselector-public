from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dataselector.cli_decorators import cli_command


@dataclass(frozen=True)
class ExplorationPlan:
    method: str
    n_initial_raw: int
    n_initial_final: int
    adjustment_rule: str
    benchmark_csv: str
    benchmark_plot: str
    n_dims: int
    n_samples_list: list[int]
    selected_score: float | None
    timestamp_utc: str
    git_commit: str | None
    python_executable: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(repo_root: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return (p.stdout or "").strip() or None
    except Exception:
        return None
    return None


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _round_up_to_multiple(n: int, m: int) -> int:
    if m <= 0:
        return n
    return int(math.ceil(n / m) * m)


def _adjust_n_initial(
    method: str, n_raw: int, *, min_sobol: int = 32, lhs_multiple: int = 8
) -> tuple[int, str]:
    method_norm = method.strip().lower()
    if method_norm in {"sobol", "qmc"}:
        adjusted = max(min_sobol, _next_power_of_two(n_raw))
        return adjusted, "sobol_power_of_two_round_up_min32"
    if method_norm in {"lhs"}:
        adjusted = _round_up_to_multiple(n_raw, lhs_multiple)
        return adjusted, f"lhs_round_up_to_multiple_{lhs_multiple}"
    # random / fallback
    return n_raw, "no_adjustment"


def _choose_best_method(
    csv_path: Path, *, prefer: str | None = None
) -> tuple[str, float | None]:
    """Choose method using benchmark CSV.

    The legacy benchmark writes rows with columns:
      - method
      - mean_time
      - std_time
      - discrepancy
      - min_distance

    We choose the method that minimizes discrepancy, using mean_time as a tie-breaker.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Benchmark CSV is empty: {csv_path}")
    # Normalize method names
    df["method_norm"] = df["method"].astype(str).str.strip().str.lower()

    if prefer:
        pref = prefer.strip().lower()
        cand = df[df["method_norm"] == pref]
        if not cand.empty:
            # still return its mean discrepancy as score
            score = (
                float(cand["discrepancy"].mean())
                if "discrepancy" in cand.columns
                else None
            )
            return pref, score

    # Filter to valid numeric discrepancy
    if "discrepancy" not in df.columns:
        raise RuntimeError(f"Benchmark CSV missing 'discrepancy' column: {csv_path}")
    cand = df[df["discrepancy"].notna()].copy()
    if cand.empty:
        raise RuntimeError(f"No valid discrepancy values in benchmark CSV: {csv_path}")

    # Aggregate per method
    agg = (
        cand.groupby("method_norm")
        .agg(discrepancy_mean=("discrepancy", "mean"), mean_time=("mean_time", "mean"))
        .reset_index()
        .sort_values(["discrepancy_mean", "mean_time"], ascending=[True, True])
    )
    best = str(agg.iloc[0]["method_norm"])
    best_score = float(agg.iloc[0]["discrepancy_mean"])
    return best, best_score


@cli_command(
    "benchmark-sampling",
    help="Benchmark initial sampling methods and persist an exploration plan",
    args={
        "n_initial_raw": {
            "type": int,
            "default": 20,
            "help": "Raw heuristic sample count before adjustment",
        },
        "n_dims": {
            "type": int,
            "default": 4,
            "help": "Dimensionality for the benchmark (default: 4)",
        },
        "n_samples": {
            "nargs": "+",
            "type": int,
            "default": [32, 64, 128],
            "help": "Sample sizes to benchmark",
        },
        "n_repeats": {
            "type": int,
            "default": 5,
            "help": "Repeats per method/sample size",
        },
        "prefer_method": {
            "type": str,
            "default": None,
            "help": "Force a specific method (sobol/lhs/random)",
        },
        "require_methods": {
            "nargs": "+",
            "default": None,
            "help": "Fail if any of these methods are missing",
        },
        "out_dir": {
            "type": str,
            "default": "outputs",
            "help": "Output directory for benchmarks",
        },
        "min_sobol": {
            "type": int,
            "default": 32,
            "help": "Minimum Sobol sample size when adjusting n_initial",
        },
        "lhs_multiple": {
            "type": int,
            "default": 8,
            "help": "Round LHS sample count to multiple of this",
        },
    },
)
def main(
    n_initial_raw: int = 20,
    n_dims: int = 4,
    n_samples: list[int] = None,
    n_repeats: int = 5,
    prefer_method: str | None = None,
    require_methods: list[str] | None = None,
    out_dir: str = "outputs",
    min_sobol: int = 32,
    lhs_multiple: int = 8,
) -> int:
    """CLI entry point for benchmark-sampling workflow."""
    # Default for n_samples
    if n_samples is None:
        n_samples = [32, 64, 128]

    # Call the core implementation logic
    return _benchmark_implementation(
        n_initial_raw=n_initial_raw,
        n_dims=n_dims,
        n_samples=n_samples,
        n_repeats=n_repeats,
        prefer_method=prefer_method,
        require_methods=require_methods,
        out_dir=out_dir,
        min_sobol=min_sobol,
        lhs_multiple=lhs_multiple,
    )


def _benchmark_implementation(
    n_initial_raw: int,
    n_dims: int,
    n_samples: list[int],
    n_repeats: int,
    prefer_method: str | None,
    require_methods: list[str] | None,
    out_dir: str,
    min_sobol: int,
    lhs_multiple: int,
) -> int:
    """Core implementation logic (extracted from old main())."""
    repo_root = _repo_root()
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Run the legacy benchmark script in a subprocess (same interpreter)
    bench_csv = out_dir_path / "sampling_benchmark_results.csv"
    bench_plot = out_dir_path / "sampling_benchmark_plots.png"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "benchmark_sampling_methods.py"),
        "--n-samples",
        *[str(x) for x in n_samples],
        "--n-dims",
        str(n_dims),
        "--n-repeats",
        str(n_repeats),
        "--output-dir",
        str(out_dir_path),
    ]

    p = subprocess.run(cmd, cwd=str(repo_root))
    if p.returncode != 0:
        return int(p.returncode)

    if not bench_csv.exists():
        raise RuntimeError(f"Benchmark did not write expected CSV: {bench_csv}")
    if not bench_plot.exists():
        # Plot may be missing if matplotlib failed; treat as error for thesis-grade usage.
        raise RuntimeError(f"Benchmark did not write expected plot: {bench_plot}")

    chosen_method, score = _choose_best_method(bench_csv, prefer=prefer_method)

    # Ensure required methods are present if requested
    if require_methods:
        import pandas as pd

        df = pd.read_csv(bench_csv)
        present = set(df["method"].astype(str).str.strip().str.lower().unique())
        missing = [
            m for m in [x.strip().lower() for x in require_methods] if m not in present
        ]
        if missing:
            raise RuntimeError(
                "Required benchmark methods missing from results: " + ", ".join(missing)
            )

    n_final, rule = _adjust_n_initial(
        chosen_method,
        int(n_initial_raw),
        min_sobol=int(min_sobol),
        lhs_multiple=int(lhs_multiple),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = ExplorationPlan(
        method=chosen_method,
        n_initial_raw=int(n_initial_raw),
        n_initial_final=int(n_final),
        adjustment_rule=rule,
        benchmark_csv=str(bench_csv),
        benchmark_plot=str(bench_plot),
        n_dims=int(n_dims),
        n_samples_list=[int(x) for x in n_samples],
        selected_score=score,
        timestamp_utc=ts,
        git_commit=_git_commit(repo_root),
        python_executable=sys.executable,
    )

    plans_dir = out_dir_path / "exploration_plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir_path / "exploration_plan_latest.json"
    stamped = plans_dir / f"exploration_plan_{ts}.json"

    payload = asdict(plan)
    latest.write_text(json.dumps(payload, indent=2))
    stamped.write_text(json.dumps(payload, indent=2))

    # Also export to environment for downstream wrappers if this subcommand is used in-process.
    os.environ["DATASELECTOR_EXPLORATION_PLAN"] = str(latest)

    print(f"Wrote exploration plan: {latest}")
    print(f"Wrote exploration plan (timestamped): {stamped}")
    print(
        f"Chosen method: {chosen_method} | n_initial: {n_initial_raw} -> "
        f"{n_final} ({rule})"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dataselector benchmark-sampling",
        description="Benchmark initial sampling methods and persist an exploration plan",
    )
    parser.add_argument(
        "--n-initial-raw", type=int, default=20, help="Raw heuristic sample count"
    )
    parser.add_argument(
        "--n-dims", type=int, default=4, help="Dimensionality for benchmark"
    )
    parser.add_argument(
        "--n-samples", nargs="+", type=int, default=[32, 64, 128], help="Sample sizes"
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5, help="Repeats per method/sample size"
    )
    parser.add_argument(
        "--prefer-method", type=str, default=None, help="Force chosen method"
    )
    parser.add_argument(
        "--require-methods", nargs="+", default=None, help="Required methods"
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--min-sobol", type=int, default=32, help="Minimum Sobol sample size"
    )
    parser.add_argument(
        "--lhs-multiple", type=int, default=8, help="Round LHS to multiple of"
    )

    args = parser.parse_args()
    sys.exit(
        main(
            n_initial_raw=args.n_initial_raw,
            n_dims=args.n_dims,
            n_samples=args.n_samples,
            n_repeats=args.n_repeats,
            prefer_method=args.prefer_method,
            require_methods=args.require_methods,
            out_dir=args.out_dir,
            min_sobol=args.min_sobol,
            lhs_multiple=args.lhs_multiple,
        )
    )
