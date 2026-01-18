#!/usr/bin/env python3
"""Analyse Bootstrap Konvergenz um optimales n_boot zu bestimmen.

Usage:
    python scripts/analyze_bootstrap_convergence.py
    python scripts/analyze_bootstrap_convergence.py --n-repeats 20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci_width(data, metric_func, n_boot, seed=42):
    """Berechne Bootstrap CI-Breite für gegebenes n_boot."""
    rng = np.random.RandomState(seed)
    boot_values = []
    
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_values.append(metric_func(sample))
    
    ci_lower, ci_upper = np.percentile(boot_values, [2.5, 97.5])
    ci_width = ci_upper - ci_lower
    
    return ci_width


def analyze_convergence(data, metric_func=np.mean, 
                        n_boot_values=None,
                        n_repeats=10, 
                        seed=42,
                        output_dir=Path('outputs')):
    """Teste Bootstrap Konvergenz für verschiedene n_boot Werte."""
    
    if n_boot_values is None:
        n_boot_values = [20, 50, 100, 200, 500, 1000]
    
    print(f"\n{'='*60}")
    print("BOOTSTRAP KONVERGENZANALYSE")
    print(f"{'='*60}")
    print(f"Datenpunkte:      {len(data)}")
    print(f"Metrik:           {metric_func.__name__}")
    print(f"Wiederholungen:   {n_repeats}")
    print(f"n_boot Werte:     {n_boot_values}")
    print(f"{'='*60}\n")
    
    results = {n: [] for n in n_boot_values}
    base_rng = np.random.RandomState(seed)
    
    for n_boot in n_boot_values:
        print(f"Testing n_boot={n_boot}...", end=' ')
        for rep in range(n_repeats):
            rep_seed = base_rng.randint(0, 10000)
            ci_width = bootstrap_ci_width(data, metric_func, n_boot, seed=rep_seed)
            results[n_boot].append(ci_width)
        print(f"✓ (mean CI width: {np.mean(results[n_boot]):.4f})")
    
    # Berechne Statistiken
    means = [np.mean(results[n]) for n in n_boot_values]
    stds = [np.std(results[n]) for n in n_boot_values]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: CI Width vs. n_boot
    ax1.errorbar(n_boot_values, means, yerr=stds, marker='o', markersize=8,
                capsize=5, capthick=2, linewidth=2, color='#2E86AB',
                ecolor='#A23B72', alpha=0.8)
    ax1.set_xlabel('Number of Bootstrap Resamples', fontsize=12, fontweight='bold')
    ax1.set_ylabel('95% CI Width (mean ± std)', fontsize=12, fontweight='bold')
    ax1.set_title('Bootstrap Konvergenz: CI-Breite vs. n_boot', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Markiere n_boot=200
    if 200 in n_boot_values:
        idx_200 = n_boot_values.index(200)
        ax1.axvline(x=200, color='#F18F01', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'n_boot=200 (aktuell)')
        ax1.plot(200, means[idx_200], 'o', markersize=12, color='#F18F01',
                markeredgecolor='black', markeredgewidth=2)
        ax1.legend(fontsize=10)
    
    # Plot 2: Relative Change in CI Width
    relative_change = [abs(means[i] - means[i-1]) / means[i-1] * 100 
                      if i > 0 else 0 
                      for i in range(len(means))]
    
    ax2.plot(n_boot_values, relative_change, marker='s', markersize=8,
            linewidth=2, color='#06A77D', alpha=0.8)
    ax2.axhline(y=5, color='#D62246', linestyle='--', linewidth=2,
               label='5% Änderung (Konvergenz-Threshold)')
    ax2.set_xlabel('Number of Bootstrap Resamples', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Change in CI Width (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Bootstrap Konvergenz: Relative Änderung', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'bootstrap_convergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("ERGEBNISSE")
    print(f"{'='*60}")
    
    df = pd.DataFrame({
        'n_boot': n_boot_values,
        'ci_width_mean': means,
        'ci_width_std': stds,
        'relative_change_%': relative_change
    })
    
    print(df.to_string(index=False))
    
    # Konvergenzanalyse
    print(f"\n{'='*60}")
    print("KONVERGENZANALYSE")
    print(f"{'='*60}")
    
    # Finde Konvergenzpunkt (< 5% Änderung)
    conv_indices = [i for i, rc in enumerate(relative_change) if rc < 5.0 and i > 0]
    
    if conv_indices:
        conv_idx = min(conv_indices)
        conv_n_boot = n_boot_values[conv_idx]
        print(f"✅ Konvergenz erreicht bei n_boot={conv_n_boot}")
        print(f"   (< 5% Änderung gegenüber vorherigem Wert)")
        
        if conv_n_boot <= 100:
            print(f"\n💡 EMPFEHLUNG: n_boot=100 ist ausreichend für stabile CI")
        elif conv_n_boot <= 200:
            print(f"\n✅ EMPFEHLUNG: Aktuelle n_boot=200 ist optimal")
        else:
            print(f"\n⚠️  EMPFEHLUNG: n_boot={conv_n_boot} oder höher verwenden")
    else:
        print(f"⚠️  Keine Konvergenz erreicht. Höhere n_boot Werte testen!")
    
    # Variabilität bei n_boot=200
    if 200 in n_boot_values:
        idx_200 = n_boot_values.index(200)
        cv = stds[idx_200] / means[idx_200] * 100  # Coefficient of Variation
        print(f"\n📊 Bei n_boot=200:")
        print(f"   CI Width: {means[idx_200]:.4f} ± {stds[idx_200]:.4f}")
        print(f"   Variationskoeffizient: {cv:.2f}%")
        
        if cv < 5:
            print(f"   ✅ Sehr stabil (CV < 5%)")
        elif cv < 10:
            print(f"   ✅ Stabil (CV < 10%)")
        else:
            print(f"   ⚠️  Hohe Variabilität (CV > 10%), mehr Resamples erwägen")
    
    print(f"\n📁 Plot gespeichert: {output_path}")
    
    # Speichere Ergebnisse
    csv_path = output_dir / 'bootstrap_convergence_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"📁 Ergebnisse gespeichert: {csv_path}")
    
    return df


def load_bootstrap_data(outputs_dir=Path('outputs')):
    """Lade existierende Bootstrap-Daten falls vorhanden."""
    
    # Suche nach validation/seed_benchmark Ergebnissen
    search_patterns = [
        'outputs/validation_seeded/*bootstrap*.csv',
        'outputs/validation_fine_seeded/*bootstrap*.csv',
        'outputs/seed_benchmark/*bootstrap*.csv',
        'outputs/*bootstrap*.csv'
    ]
    
    for pattern in search_patterns:
        files = list(Path('.').glob(pattern))
        if files:
            print(f"Gefunden: {files[0]}")
            df = pd.read_csv(files[0])
            
            # Versuche relevante Spalte zu finden
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                print(f"Verwende Spalte: {col}")
                return df[col].values
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Analysiere Bootstrap Konvergenz für n_boot Optimierung'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=10,
        help='Anzahl Wiederholungen pro n_boot (default: 10)'
    )
    parser.add_argument(
        '--n-boot-values',
        type=int,
        nargs='+',
        default=[20, 50, 100, 200, 500, 1000],
        help='Liste von n_boot Werten zum Testen'
    )
    parser.add_argument(
        '--data-file',
        type=Path,
        help='CSV-Datei mit Bootstrap-Daten (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output Verzeichnis (default: outputs/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lade oder generiere Testdaten
    if args.data_file and args.data_file.exists():
        print(f"Lade Daten aus: {args.data_file}")
        df = pd.read_csv(args.data_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print("❌ Keine numerischen Spalten gefunden!")
            return
        data = df[numeric_cols[0]].values
        print(f"Verwende Spalte: {numeric_cols[0]} ({len(data)} Datenpunkte)")
    
    else:
        # Versuche existierende Bootstrap-Daten zu laden
        data = load_bootstrap_data()
        
        if data is None:
            # Generiere synthetische Testdaten
            print("⚠️  Keine existierenden Bootstrap-Daten gefunden.")
            print("Generiere synthetische Testdaten für Demonstration...")
            rng = np.random.RandomState(args.seed)
            data = rng.randn(100) * 10 + 50  # N(50, 10)
            print(f"Synthetische Daten: N={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}")
    
    # Führe Konvergenzanalyse durch
    analyze_convergence(
        data=data,
        metric_func=np.mean,
        n_boot_values=args.n_boot_values,
        n_repeats=args.n_repeats,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
