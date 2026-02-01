<<<<<<< HEAD
"""Plots bootstrap summary for Pareto candidates."""

from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fine_sweep"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Helper: attempt to load canonical summary, otherwise search run-level summaries and normalize
    def _load_bootstrap_summary() -> pd.DataFrame:
        # Primary canonical location
        main_file = OUT / "bootstrap_results_summary.csv"
        if main_file.exists():
            return pd.read_csv(main_file)
        # Search run-level summaries
        from glob import glob

        candidates = list(Path("outputs").glob("**/*bootstrap*summary*.csv"))
        if not candidates:
            raise FileNotFoundError("No bootstrap summary found in outputs or runs")

        dfs = []
        for p in candidates:
            try:
                d = pd.read_csv(p)
                # If this is a per-iteration table, try to aggregate
                if not any(col.endswith("_mean") for col in d.columns):
                    group_cols = [c for c in ["alpha", "beta", "gamma", "min_distance_km"] if c in d.columns]
                    metric_cols = [c for c in d.columns if c not in group_cols]
                    if group_cols and metric_cols:
                        aggs = {c: ["mean", "std"] for c in metric_cols if d[c].dtype.kind in "fi"}
                        if aggs:
                            d = d.groupby(group_cols).agg(aggs).reset_index()
                            # Flatten
                            d.columns = ["_".join([str(i) for i in col]).strip("_") if isinstance(col, tuple) else col for col in d.columns]
                dfs.append(d)
            except Exception:
                continue
        if not dfs:
            raise ValueError("Found summary files but none were readable as CSV")
        df = pd.concat(dfs, ignore_index=True, sort=False)
        # Normalize column aliases for plotting
        # prefer temporal_std_mean / temporal_std_std
        if "temporal_std_mean" not in df.columns and "temporal_std_mean" not in df.columns:
            for col in df.columns:
                if col.lower().startswith("temporal_std") and "mean" in col.lower():
                    df = df.rename(columns={col: "temporal_std_mean"})
                if col.lower().startswith("temporal_std") and "std" in col.lower():
                    df = df.rename(columns={col: "temporal_std_std"})
        # wwi
        if "wwi_percent_mean" not in df.columns:
            for col in df.columns:
                if "wwi" in col.lower() and "mean" in col.lower():
                    df = df.rename(columns={col: "wwi_percent_mean"})
                if "wwi" in col.lower() and "std" in col.lower():
                    df = df.rename(columns={col: "wwi_percent_std"})
        # jaccard
        if "jaccard_with_original_mean" not in df.columns and "jaccard_mean" not in df.columns:
            for col in df.columns:
                if "jaccard" in col.lower() and "mean" in col.lower():
                    df = df.rename(columns={col: "jaccard_with_original_mean"})
                if "jaccard" in col.lower() and "std" in col.lower():
                    df = df.rename(columns={col: "jaccard_with_original_std"})
        # Fallback: if jaccard_mean exists but not jaccard_with_original_mean
        if "jaccard_mean" in df.columns and "jaccard_with_original_mean" not in df.columns:
            df["jaccard_with_original_mean"] = df["jaccard_mean"]
        if "jaccard_std" in df.columns and "jaccard_with_original_std" not in df.columns:
            df["jaccard_with_original_std"] = df["jaccard_std"]
        return df

    try:
        df = _load_bootstrap_summary()
    except Exception as e:
        print("No usable bootstrap summary found for plotting:", e)
        raise

    # Prepare labels
    def _safe_fmt(x, key, fmt="{:.2f}"):
        try:
            return fmt.format(float(x.get(key, 0)))
        except Exception:
            return "?"

    def _safe_int(x, default=0):
        try:
            xv = float(x)
            if pd.isna(xv):
                return default
            return int(xv)
        except Exception:
            return default

    labels = [
        f"{_safe_fmt(r,'alpha')}/{_safe_fmt(r,'beta')}/d{_safe_int(r.get('min_distance_km',0))}"
        for _, r in df.iterrows()
    ]

    # Ensure plot directory
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    # Temporal STD
    x = np.arange(len(df))
    plt.figure(figsize=(7, 4))
    y = df.get("temporal_std_mean")
    yerr = df.get("temporal_std_std")
    if y is None:
        raise ValueError("temporal_std_mean column missing for plotting")
    plt.bar(x, y, yerr=yerr if yerr is not None else None, capsize=5)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Temporal STD (years)")
    plt.title("Bootstrap: Temporal STD (mean ± std) for Pareto candidates")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "bootstrap_temporal_std.png")

    # WWI percent
    plt.figure(figsize=(7, 4))
    y = df.get("wwi_percent_mean")
    yerr = df.get("wwi_percent_std")
    if y is not None:
        plt.bar(x, y, yerr=yerr if yerr is not None else None, capsize=5, color="orange")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("WWI %")
        plt.title("Bootstrap: WWI % (mean ± std) for Pareto candidates")
        plt.tight_layout()
        plt.savefig(OUT / "plots" / "bootstrap_wwi_percent.png")

    # Jaccard
    plt.figure(figsize=(7, 4))
    if "jaccard_with_original_mean" in df.columns:
        y = df["jaccard_with_original_mean"]
        yerr = df.get("jaccard_with_original_std")
    elif "jaccard_mean" in df.columns:
        y = df["jaccard_mean"]
        yerr = df.get("jaccard_std")
    else:
        y = None
        yerr = None

    if y is not None:
        plt.bar(x, y, yerr=yerr if yerr is not None else None, capsize=5, color="green")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Jaccard with original selection")
        plt.ylim(0, 1)
        plt.title("Bootstrap: Selection stability (Jaccard)")
        plt.tight_layout()
        plt.savefig(OUT / "plots" / "bootstrap_jaccard.png")

    print("Plots saved to", OUT / "plots")
=======
"""Plots bootstrap summary for Pareto candidates.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'fine_sweep'
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    df = pd.read_csv(OUT / 'bootstrap_results_summary.csv')
    labels = [f"{r['alpha']:.2f}/{r['beta']:.2f}/d{int(r['min_distance_km'])}" for _, r in df.iterrows()]

    # Temporal STD
    x = np.arange(len(df))
    plt.figure(figsize=(7,4))
    plt.bar(x, df['temporal_std_mean'], yerr=df['temporal_std_std'], capsize=5)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Temporal STD (years)')
    plt.title('Bootstrap: Temporal STD (mean ± std) for Pareto candidates')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'bootstrap_temporal_std.png')

    # WWI percent
    plt.figure(figsize=(7,4))
    plt.bar(x, df['wwi_percent_mean'], yerr=df['wwi_percent_std'], capsize=5, color='orange')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('WWI %')
    plt.title('Bootstrap: WWI % (mean ± std) for Pareto candidates')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'bootstrap_wwi_percent.png')

    # Jaccard
    plt.figure(figsize=(7,4))
    plt.bar(x, df['jaccard_mean'], yerr=df['jaccard_std'], capsize=5, color='green')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Jaccard with original selection')
    plt.ylim(0,1)
    plt.title('Bootstrap: Selection stability (Jaccard)')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'bootstrap_jaccard.png')

    print('Plots saved to', OUT / 'plots')
>>>>>>> ci/add-smoke-tests
