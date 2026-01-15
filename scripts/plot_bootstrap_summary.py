"""Plots bootstrap summary for Pareto candidates.
"""
from pathlib import Path
import pandas as pd
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