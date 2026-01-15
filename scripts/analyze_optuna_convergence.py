#!/usr/bin/env python3
"""Analyse Optuna Konvergenz um optimales n_trials zu bestimmen.

Usage:
    python scripts/analyze_optuna_convergence.py outputs/optuna_comparison/
    python scripts/analyze_optuna_convergence.py --study-db optuna_study.db
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_convergence_from_csv(csv_path: Path, output_dir: Path):
    """Analysiere Konvergenz aus CSV-Datei."""
    df = pd.read_csv(csv_path)
    
    if 'value' not in df.columns:
        print(f"Warnung: {csv_path} hat keine 'value' Spalte")
        return None
    
    # Sortiere nach trial number falls vorhanden
    if 'number' in df.columns:
        df = df.sort_values('number')
    
    best_values = df['value'].values
    cumulative_best = np.maximum.accumulate(best_values)
    
    # Berechne Konvergenzpunkt (99% vom Maximum)
    threshold = cumulative_best[-1] * 0.99
    conv_indices = np.where(cumulative_best >= threshold)[0]
    conv_idx = conv_indices[0] if len(conv_indices) > 0 else len(cumulative_best) - 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_best, linewidth=2, label='Best Value', color='#2E86AB')
    ax.axhline(y=threshold, color='#A23B72', linestyle='--', linewidth=1.5,
               label=f'99% von Maximum ({threshold:.3f})')
    ax.axvline(x=conv_idx, color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Konvergenz bei Trial {conv_idx}')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Best Objective Value', fontsize=12)
    ax.set_title(f'Optuna Konvergenzanalyse: {csv_path.stem}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / f"{csv_path.stem}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Analyse: {csv_path.name}")
    print(f"{'='*60}")
    print(f"Gesamt Trials:        {len(best_values)}")
    print(f"Finales Best Value:   {cumulative_best[-1]:.4f}")
    print(f"99% Threshold:        {threshold:.4f}")
    print(f"Konvergenz bei Trial: {conv_idx} ({conv_idx/len(best_values)*100:.1f}%)")
    
    if conv_idx < len(best_values) * 0.5:
        print(f"✅ EMPFEHLUNG: Konvergenz früh erreicht! n_trials könnte reduziert werden.")
    elif conv_idx < len(best_values) * 0.8:
        print(f"✅ EMPFEHLUNG: Aktuelle n_trials={len(best_values)} ist angemessen.")
    else:
        print(f"⚠️  WARNUNG: Konvergenz spät erreicht! Mehr Trials empfohlen.")
    
    print(f"Plot gespeichert: {output_path}")
    
    return {
        'csv_path': str(csv_path),
        'n_trials': len(best_values),
        'best_value': cumulative_best[-1],
        'convergence_trial': conv_idx,
        'convergence_ratio': conv_idx / len(best_values)
    }


def plot_convergence_from_study(study_path: Path, output_dir: Path):
    """Analysiere Konvergenz aus Optuna Study (pickle)."""
    try:
        import optuna
    except ImportError:
        print("Fehler: optuna nicht installiert. Bitte installieren: pip install optuna")
        return None
    
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
    
    trials = study.trials
    best_values = [t.value for t in trials if t.value is not None]
    
    if len(best_values) == 0:
        print(f"Warnung: Keine validen Trials in {study_path}")
        return None
    
    cumulative_best = np.maximum.accumulate(best_values)
    threshold = cumulative_best[-1] * 0.99
    conv_indices = np.where(cumulative_best >= threshold)[0]
    conv_idx = conv_indices[0] if len(conv_indices) > 0 else len(cumulative_best) - 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_best, linewidth=2, label='Best Value', color='#2E86AB')
    ax.axhline(y=threshold, color='#A23B72', linestyle='--', linewidth=1.5,
               label=f'99% von Maximum ({threshold:.3f})')
    ax.axvline(x=conv_idx, color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Konvergenz bei Trial {conv_idx}')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Best Objective Value', fontsize=12)
    ax.set_title(f'Optuna Konvergenzanalyse: {study.study_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / f"{study.study_name}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Analyse: {study.study_name}")
    print(f"{'='*60}")
    print(f"Gesamt Trials:        {len(best_values)}")
    print(f"Finales Best Value:   {cumulative_best[-1]:.4f}")
    print(f"99% Threshold:        {threshold:.4f}")
    print(f"Konvergenz bei Trial: {conv_idx} ({conv_idx/len(best_values)*100:.1f}%)")
    
    if conv_idx < len(best_values) * 0.5:
        print(f"✅ EMPFEHLUNG: Konvergenz früh erreicht! n_trials könnte reduziert werden.")
    elif conv_idx < len(best_values) * 0.8:
        print(f"✅ EMPFEHLUNG: Aktuelle n_trials={len(best_values)} ist angemessen.")
    else:
        print(f"⚠️  WARNUNG: Konvergenz spät erreicht! Mehr Trials empfohlen.")
    
    print(f"Plot gespeichert: {output_path}")
    
    return {
        'study_name': study.study_name,
        'n_trials': len(best_values),
        'best_value': cumulative_best[-1],
        'convergence_trial': conv_idx,
        'convergence_ratio': conv_idx / len(best_values)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analysiere Optuna Konvergenz für n_trials Optimierung'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Pfad zu Optuna Study (pickle) oder Verzeichnis mit CSVs'
    )
    parser.add_argument(
        '--study-db',
        type=str,
        help='Optuna Study Database (sqlite)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output Verzeichnis für Plots (default: outputs/)'
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    
    if args.study_db:
        # Load from database
        try:
            import optuna
            study = optuna.load_study(study_name='kdr100_opt', storage=f'sqlite:///{args.study_db}')
            result = plot_convergence_from_study(Path(args.study_db), args.output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Fehler beim Laden der Study DB: {e}")
    
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            if input_path.suffix == '.pkl':
                result = plot_convergence_from_study(input_path, args.output_dir)
                if result:
                    results.append(result)
            elif input_path.suffix == '.csv':
                result = plot_convergence_from_csv(input_path, args.output_dir)
                if result:
                    results.append(result)
        
        elif input_path.is_dir():
            # Suche alle CSV-Dateien im Verzeichnis
            csv_files = sorted(input_path.glob('*.csv'))
            pkl_files = sorted(input_path.glob('*.pkl'))
            
            for csv_file in csv_files:
                result = plot_convergence_from_csv(csv_file, args.output_dir)
                if result:
                    results.append(result)
            
            for pkl_file in pkl_files:
                result = plot_convergence_from_study(pkl_file, args.output_dir)
                if result:
                    results.append(result)
    
    else:
        # Default: Suche in outputs/
        print("Suche Optuna-Ergebnisse in outputs/...")
        search_dirs = [
            Path('outputs/optuna_comparison'),
            Path('outputs/optuna_comparison_v2'),
            Path('outputs/optuna_comparison_v3'),
            Path('outputs'),
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            csv_files = sorted(search_dir.glob('*trials*.csv'))
            for csv_file in csv_files:
                result = plot_convergence_from_csv(csv_file, args.output_dir)
                if result:
                    results.append(result)
    
    # Zusammenfassung
    if results:
        print(f"\n{'='*60}")
        print("GESAMTZUSAMMENFASSUNG")
        print(f"{'='*60}")
        
        df_summary = pd.DataFrame(results)
        print(df_summary.to_string(index=False))
        
        summary_path = args.output_dir / 'optuna_convergence_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"\nZusammenfassung gespeichert: {summary_path}")
        
        # Gesamtempfehlung
        avg_conv_ratio = df_summary['convergence_ratio'].mean()
        print(f"\n📊 Durchschnittliche Konvergenz bei {avg_conv_ratio*100:.1f}% der Trials")
        
        if avg_conv_ratio < 0.5:
            print("✅ EMPFEHLUNG: n_trials könnte auf ~100-150 reduziert werden")
        elif avg_conv_ratio < 0.8:
            print("✅ EMPFEHLUNG: Aktuelle n_trials=200 ist optimal")
        else:
            print("⚠️  EMPFEHLUNG: n_trials auf 300-500 erhöhen für bessere Konvergenz")
    
    else:
        print("\n❌ Keine Optuna-Ergebnisse gefunden.")
        print("Bitte Pfad angeben oder Optuna-Experimente durchführen.")


if __name__ == '__main__':
    main()
