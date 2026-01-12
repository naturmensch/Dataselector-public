"""Plot comparison between seeded and unseeded validation summaries.
Saves temporal_std and wwi_percent comparison plots to outputs/validation/plots.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "validation" / "seeded_vs_unseeded_summary.csv"
OUT = ROOT / "outputs" / "validation" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# For readability, create a label for each weight combo
df['combo'] = df.apply(lambda r: f"a{r['alpha']}_b{r['beta']}_g{r['gamma']}", axis=1)

# Plot temporal_std (seeded vs unseeded) as grouped bar (per min_distance)
for min_d in sorted(df['min_distance_km'].unique()):
    sub = df[df['min_distance_km']==min_d]
    x = range(len(sub))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x, sub['temporal_std_unseeded'], width=0.4, label='unseeded', align='center')
    ax.bar([i+0.4 for i in x], sub['temporal_std_seeded'], width=0.4, label='seeded', align='center')
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(sub['combo'], rotation=45, ha='right')
    ax.set_ylabel('Temporal STD (years)')
    ax.set_title(f'Temporal STD: Seeded vs Unseeded (min_dist={min_d} km)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / f'temporal_std_seeded_vs_unseeded_d{int(min_d)}.png', dpi=300)
    plt.close()

# Plot WWI percent
for min_d in sorted(df['min_distance_km'].unique()):
    sub = df[df['min_distance_km']==min_d]
    x = range(len(sub))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x, sub['wwi_percent_unseeded'], width=0.4, label='unseeded', align='center')
    ax.bar([i+0.4 for i in x], sub['wwi_percent_seeded'], width=0.4, label='seeded', align='center')
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(sub['combo'], rotation=45, ha='right')
    ax.set_ylabel('WWI percent (%)')
    ax.set_title(f'WWI %: Seeded vs Unseeded (min_dist={min_d} km)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / f'wwi_percent_seeded_vs_unseeded_d{int(min_d)}.png', dpi=300)
    plt.close()

print('Plots written to', OUT)