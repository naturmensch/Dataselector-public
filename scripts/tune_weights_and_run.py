"""
Phase 1: Exploration & Visualization (LHS Sweep).

Generiert Gewichtungen mittels Latin Hypercube Sampling (LHS),
um die Pareto-Front und Trade-offs für die Thesis zu visualisieren.

Warum LHS?
- Deterministische Abdeckung ohne Lücken im Parameterraum
- Perfekt für Thesis-Plots: zeigt Trade-off-Kurven, nicht nur einen Punkt
- Garantiert stratifizierte Abdeckung in allen Dimensionen

Usage:
    PYTHONPATH=. python scripts/tune_weights_and_run.py
    PYTHONPATH=. python scripts/tune_weights_and_run.py --n-samples 100
"""

from pathlib import Path
import sys
import argparse
import numpy as np

# Error-handling für scipy.stats.qmc (scipy>=1.7.0)
try:
    from scipy.stats import qmc
    HAS_LHS = True
except ImportError:
    print("⚠️  scipy.stats.qmc nicht gefunden. Fallback auf manuelles Grid...")
    HAS_LHS = False

# Config
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_META = ROOT / "data" / "new_all_tiles.csv"
OUTPUT_DIR = ROOT / "outputs" / "tuning_weights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter für die Exploration
MIN_DISTANCE_KM = 28.0  # Median des Datensatzes (wissenschaftlich begründet)


# --- WISSENSCHAFTLICHE PARAMETER-GENERIERUNG (LHS) ---


from src.sampling_strategies import (
    sample_weights_on_simplex_lhs,
    sample_weights_on_simplex_sobol,
)


def generate_weights(n_points: int = 50, seed: int = 42, sampler: str = 'lhs'):
    """
    Generate weight combinations on the simplex using the specified sampler.

    Args:
        n_points: number of samples
        seed: random seed
        sampler: 'lhs' or 'sobol'

    Returns:
        List of (alpha, beta, gamma) tuples summing to 1.0
    """
    if sampler.lower() == 'sobol':
        if not hasattr(sample_weights_on_simplex_sobol, '__call__'):
            raise RuntimeError("Sobol sampler not available")
        return sample_weights_on_simplex_sobol(n_points, dim=3, seed=seed)

    # default to LHS behavior (existing behavior)
    if sampler.lower() == 'lhs':
        return sample_weights_on_simplex_lhs(n_points, dim=3, seed=seed)

    raise ValueError(f"Unknown sampler: {sampler}")


# --- ENDE GENERIERUNG ---


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Exploration mit LHS-Sweep für Thesis-Plots"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Anzahl Samples (default: 50)",
    )
    parser.add_argument(
        "--sampler",
        choices=['lhs', 'sobol'],
        default='lhs',
        help="Sampler für die Generierung der Gewichtungen (lhs=LatinHypercube, sobol=QMC Sobol), default: lhs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=MIN_DISTANCE_KM,
        help=f"Min distance constraint in km (default: {MIN_DISTANCE_KM})",
    )
    parser.add_argument("--pre-names", type=str, nargs="*", default=None, help="Optional pre-selected tile names (e.g. 'Hamburg')")
    parser.add_argument("--pre-indices", type=int, nargs="*", default=None, help="Optional pre-selected tile indices (zero-based)")
    args = parser.parse_args()

    # Pipeline integration: attach to existing ExperimentManager if EXPERIMENT_RUN_DIR is set
    import os
    em = None
    exp_dir = os.environ.get('EXPERIMENT_RUN_DIR')
    if exp_dir:
        from src.experiment_manager import ExperimentManager
        em = ExperimentManager.from_existing(exp_dir)
        em.log('Attached to pipeline run (exploration stage)')
        em.save_config('exploration', {'n_samples': args.n_samples, 'sampler': args.sampler, 'seed': args.seed})

    try:
        from src.experiments import ExperimentRunner
        from src.pareto import (
            compute_pareto_front,
            export_pareto_report,
            visualize_pareto_front,
        )
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("Stelle sicher, dass alle src-Module verfügbar sind.")
        sys.exit(1)

    runner = ExperimentRunner(output_dir=str(OUTPUT_DIR))

    # Generiere Gewicht-Kombinationen (wahlweise LHS oder Sobol)
    weight_combinations = generate_weights(
        n_points=args.n_samples, seed=args.seed, sampler=args.sampler
    )

    print("\n" + "=" * 70)
    print("PHASE 1: EXPLORATION (LHS SWEEP)")
    print("=" * 70)
    print(f"Gewicht-Kombinationen: {len(weight_combinations)} (LHS-Samples)")
    print(f"Min Distance Constraint: {args.min_distance} km")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

    # Run full sweep mit LHS-Gewichten
    try:
        results = runner.run_weight_sweep(
            csv_meta=str(DATA_META),
            n_samples=673,  # Volle Datensatz-Größe
            weight_combinations=weight_combinations,  # Jetzt automatisch generiert
            n_clusters=8,
            batch_size=16,
            min_distance_km=args.min_distance,
            patience=None,  # Kein Early-Stopping, wir wollen die ganze Front sehen
            pre_selected=args.pre_indices,
            pre_selected_names=args.pre_names,
        )
    except Exception as e:
        print(f"❌ Fehler beim Ausführen des Weight Sweep: {e}")
        sys.exit(1)

    # Compute Pareto front
    print("\n" + "=" * 70)
    print("COMPUTING PARETO-FRONT (Exploration Phase)...")
    print("=" * 70)

    try:
        pareto_front = compute_pareto_front(results)
        print(
            f"✅ Pareto-Front: {len(pareto_front)} von {len(results)} "
            "Lösungen sind Pareto-optimal\n"
        )

        # Visualisierungen speichern (wichtig für Thesis!)
        if em is not None:
            viz_dir = em.get_path('artifacts') / "pareto"
            viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            viz_dir = OUTPUT_DIR / "pareto"
            viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"Erstelle Visualisierungen in {viz_dir}...")
        visualize_pareto_front(results, pareto_front, output_dir=str(viz_dir))

        # Report exportieren
        report_path = viz_dir / "pareto_solutions.csv"
        export_pareto_report(pareto_front, output_path=str(report_path))

        # Save into ExperimentManager if attached
        try:
            import pandas as _pd
            if em is not None:
                em.save_results('pareto_solutions', _pd.read_csv(report_path), format='csv')
                em.mark_stage_complete('exploration', summary={'pareto_count': len(pareto_front), 'n_samples': args.n_samples})
        except Exception as e:
            print(f"Warning: could not save pareto to experiment manager: {e}")

        print(f"\n✅ Phase 1 ABGESCHLOSSEN")
        print(f"📊 Plots: {viz_dir}")
        print(f"📋 CSV:   {report_path}")

    except Exception as e:
        print(f"❌ Fehler bei Pareto-Berechnung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
