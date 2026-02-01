#!/usr/bin/env python3
"""
Orchestrator for thesis-grade sampler evaluation.

1) Run multi-seed sampler comparisons on 'hamburg' and 'kdr100'
2) Compute best sampler per dataset and overall
3) Launch full adaptive runs (n_trials=2000) with best sampler on Hamburg and KDR100

Usage:
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def choose_best_sampler(results_dir: Path):
    df_all = []
    for s in summaries:
        try:
            df = pd.read_csv(s)
    run_cmd(compare_cmd)

    # 2) Choose best sampler
    try:
        best, table = choose_best_sampler(suite_dir)
        print(f"Best sampler (overall mean of dataset summaries): {best}")
        (suite_dir / "best_sampler_summary.json").write_text(
            json.dumps(
                {"best": best, "summary_table": table.to_dict(orient="records")},
                indent=2,
            )
        )
=======
>>>>>>> chore/ci-lint-attrs-gdf
    except Exception as e:
        print(f"ERROR selecting best sampler: {e}")
        sys.exit(1)

    # 3) Launch full adaptive runs with best sampler: Hamburg and KDR100 (no --hamburg == full)
    # Use exec_in_env.sh wrapper if available

    # Hamburg full run
    run_name_h = f"suite_full_{best}_hamburg_{timestamp}"
    cmd_h = f"PYTHONPATH=. {wrapper} --env dataselector -- python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {args.n_candidates} --sampler {best} --seed {args.seeds[0]} --hamburg"
    print(f"Launching full Hamburg run: {cmd_h}")
    run_cmd(cmd_h)

    # KDR100 full run (no preselection)
    run_name_k = f"suite_full_{best}_kdr100_{timestamp}"
    cmd_k = f"PYTHONPATH=. {wrapper} --env dataselector -- python scripts/run_adaptive_pipeline.py --yes --n-trials {args.n_trials_full} --n-candidates {args.n_candidates} --sampler {best} --seed {args.seeds[0]}"
    print(f"Launching full KDR100 run: {cmd_k}")
    run_cmd(cmd_k)

