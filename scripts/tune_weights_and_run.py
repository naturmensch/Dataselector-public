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


def generate_lhs_weights(n_points: int = 50, seed: int = 42):
    """
    Generiert Gewichtungen mittels Latin Hypercube Sampling auf dem Simplex.

    Args:
        n_points: Anzahl der LHS-Samples
        seed: Random seed für Reproduzierbarkeit

    Returns:
        Liste von (alpha, beta, gamma) Tuples mit Σ = 1.0
    """
    if not HAS_LHS:
        # Fallback: Manuelles Grid (9 Kombinationen)
        print("⚠️  Fallback: Verwende predefiniertes Grid statt LHS")
        return [
            (0.70, 0.15, 0.15),
            (0.70, 0.20, 0.10),
            (0.75, 0.15, 0.10),
            (0.60, 0.20, 0.20),
            (0.65, 0.20, 0.15),
            (0.60, 0.15, 0.25),
            (0.55, 0.25, 0.20),
            (0.50, 0.30, 0.20),
            (0.65, 0.25, 0.10),
        ]

    print(f"Generiere {n_points} LHS-Samples für Gewichte...")

    # 1. Erzeuge LHS Samples im 3D-Hyperwürfel [0,1]^3
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    sample = sampler.random(n=n_points)

    # 2. Projiziere auf den Simplex (Summe = 1) durch Normalisierung
    # Dies ist eine Standardmethode für Random Search über Gewichtungen
    weights = sample / sample.sum(axis=1)[:, None]

    return [tuple(w) for w in weights]


# --- ENDE GENERIERUNG ---


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Exploration mit LHS-Sweep für Thesis-Plots"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Anzahl LHS-Samples (default: 50)",
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
    args = parser.parse_args()

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

    # Generiere LHS-Kombinationen
    weight_combinations = generate_lhs_weights(
        n_points=args.n_samples, seed=args.seed
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
        viz_dir = OUTPUT_DIR / "pareto"
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"Erstelle Visualisierungen in {viz_dir}...")
        visualize_pareto_front(results, pareto_front, output_dir=str(viz_dir))

        # Report exportieren
        report_path = viz_dir / "pareto_solutions.csv"
        export_pareto_report(pareto_front, output_path=str(report_path))

        print(f"\n✅ Phase 1 ABGESCHLOSSEN")
        print(f"📊 Plots: {viz_dir}")
        print(f"📋 CSV:   {report_path}")

    except Exception as e:
        print(f"❌ Fehler bei Pareto-Berechnung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
