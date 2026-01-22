"""
Master-Pipeline für die Thesis-Optimierung.

Führt 4 Phasen sequenziell aus:
  1. EXPLORATION (LHS): Pareto-Front visualisieren
  2. OPTIMIZATION (Optuna): Bayesian-optimierte Parameter finden
  3. VALIDATION (Bootstrap): Robustheit der Pareto-Kandidaten testen
  4. SUMMARY: Report & Vergleich

Wissenschaftliche Begründung:
- LHS + Optuna = "Hybrid-Approach"
- Exploration zeigt Trade-offs (für Thesis-Plots)
- Optimization findet Optimum (für beste Ergebnisse)
- Validation beweist Robustheit (für publikationsfähigkeit)

Usage:
    python scripts/run_thesis_pipeline.py
    python scripts/run_thesis_pipeline.py --n-lhs 100 --skip-optimization
    python scripts/run_thesis_pipeline.py --help
"""

import argparse
import glob
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# NOTE: Do not perform dataset I/O at import time. The thesis default for `n_lhs`
# will be computed inside `main()` if the user does not provide `--n-lhs`.
n_lhs_thesis_default = 50  # conservative fallback if metadata cannot be read at runtime


def run_step(step_name: str, command: list, skip: bool = False):
    """Führe einen Pipeline-Schritt aus mit Fehlerbehandlung."""
    if skip:
        print(f"\n⏭️  ÜBERSPRINGE {step_name}")
        return True

    print("\n" + "#" * 80)
    print(f"### STARTE {step_name}")
    print("#" * 80)

    t0 = time.time()
    try:
        print(f"Kommando: {' '.join(command)}\n")
        subprocess.check_call(command)
        elapsed = time.time() - t0
        print(f"\n✅ {step_name} erfolgreich (Dauer: {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER in {step_name}: Exit Code {e.returncode}")
        print(f"Kommando war: {' '.join(command)}")
        return False
    except Exception as e:
        print(f"\n❌ FEHLER in {step_name}: {e}")
        return False


def find_latest_optuna_file():
    """Finde automatisch die neueste Optuna-Resultatdatei."""
    optuna_files = sorted(glob.glob(str(OUTPUT_DIR / "optuna_autoscale_best_*.json")))
    if optuna_files:
        latest = optuna_files[-1]
        print(f"✅ Gefunden: {Path(latest).name}")
        return latest
    else:
        print("⚠️  Keine Optuna-Resultatdatei gefunden!")
        return None


def find_latest_pareto_file():
    """Finde automatisch die neueste Pareto-CSV-Datei."""
    pareto_files = sorted(
        glob.glob(
            str(OUTPUT_DIR / "tuning_weights" / "pareto" / "pareto_solutions.csv")
        )
    )
    if pareto_files:
        latest = pareto_files[-1]
        print(f"✅ Gefunden: {Path(latest).name}")
        return latest
    else:
        print("⚠️  Keine Pareto-Solutions-Datei gefunden!")
        return None


def load_optuna_best(optuna_file: str):
    """Lade beste Parameter aus Optuna-JSON."""
    try:
        with open(optuna_file, "r") as f:
            data = json.load(f)
            return data.get("user_attrs", {})
    except Exception as e:
        print(f"⚠️  Konnte Optuna-Datei nicht lesen: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Master-Pipeline: Exploration → Optimization → Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Volle Pipeline (Standard)
  python scripts/run_thesis_pipeline.py

  # Nur Exploration + Optimization (keine Validation)
  python scripts/run_thesis_pipeline.py --skip-validation

  # Mit 100 LHS-Samples statt 50
  python scripts/run_thesis_pipeline.py --n-lhs 100

  # Nur Validation (gehe davon aus, dass Phase 1-3 bereits fertig sind)
  python scripts/run_thesis_pipeline.py --skip-exploration --skip-optimization
        """,
    )
    # Note: default is None; compute adaptive default at runtime to avoid import-time I/O
    parser.add_argument(
        "--n-lhs",
        type=int,
        default=None,
        help=f"Anzahl LHS-Samples für Phase 1 (computed from dataset if omitted; fallback: {n_lhs_thesis_default})",
    )
    parser.add_argument(
        "--skip-exploration",
        action="store_true",
        help="Überspringe Phase 1 (LHS Sweep)",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Überspringe Phase 2/3 (Optuna)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Überspringe Phase 4 (Validation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige Kommandos ohne sie auszuführen",
    )

    args = parser.parse_args()

    # If user didn't provide n_lhs, compute an adaptive default now (deferred I/O)
    if args.n_lhs is None:
        try:
            import pandas as pd
            import numpy as np

            metadata_path = ROOT / "data" / "new_all_tiles.csv"
            if metadata_path.exists():
                n_tiles = len(pd.read_csv(metadata_path))
                args.n_lhs = max(50, int(2 * np.sqrt(n_tiles)))
                print(f"📊 Adaptive n_lhs computed from dataset: {args.n_lhs}")
            else:
                args.n_lhs = n_lhs_thesis_default
                print(f"⚠️ Metadata not found; using fallback n_lhs={args.n_lhs}")
        except Exception:
            args.n_lhs = n_lhs_thesis_default
            print(f"⚠️ Could not compute adaptive n_lhs; using fallback n_lhs={args.n_lhs}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("🚀 THESIS OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Start: {timestamp}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    all_success = True

    # Phase 1: Exploration (LHS Sweep)
    print("\n" + "=" * 80)
    print("PHASE 1: EXPLORATION (LHS-based Pareto-Front)")
    print("=" * 80)
    cmd_phase1 = [
        sys.executable,
        str(ROOT / "scripts" / "tune_weights_and_run.py"),
        "--n-samples",
        str(args.n_lhs),
    ]
    if args.dry_run:
        print(f"[DRY-RUN] würde ausführen: {' '.join(cmd_phase1)}")
    else:
        if not run_step(
            "Phase 1: Exploration (LHS)", cmd_phase1, args.skip_exploration
        ):
            all_success = False
            if not args.skip_optimization and not args.skip_validation:
                print("⚠️  Phase 1 fehlgeschlagen, aber fahre fort...")

    # Phase 2/3: Optimization (Optuna Autoscale)
    print("\n" + "=" * 80)
    print("PHASE 2/3: OPTIMIZATION (Bayesian mit Optuna)")
    print("=" * 80)
    cmd_phase23 = [
        sys.executable,
        str(ROOT / "scripts" / "optuna_autoscale.py"),
        "--stages",
        "50",
        "100",
        "300",
        "full",
        "--seed",
        "42",
    ]
    if args.dry_run:
        print(f"[DRY-RUN] würde ausführen: {' '.join(cmd_phase23)}")
    else:
        if not run_step(
            "Phase 2/3: Optimization (Optuna)", cmd_phase23, args.skip_optimization
        ):
            all_success = False
            if not args.skip_validation:
                print("⚠️  Phase 2/3 fehlgeschlagen, aber fahre fort...")

    # Phase 4: Validation (Bootstrap)
    print("\n" + "=" * 80)
    print("PHASE 4: VALIDATION (Robustheit & Sensitivität)")
    print("=" * 80)

    # Versuche, Pareto-Datei zu finden
    pareto_file = find_latest_pareto_file()
    if pareto_file and not args.skip_validation:
        cmd_phase4 = [
            sys.executable,
            str(ROOT / "scripts" / "validate_pareto_candidates.py"),
            "--pareto",
            pareto_file,
            "--seeds",
            "42",
            "43",
            "44",
            "45",
            "46",
        ]
        if args.dry_run:
            print(f"[DRY-RUN] würde ausführen: {' '.join(cmd_phase4)}")
        else:
            if not run_step("Phase 4: Validation", cmd_phase4, args.skip_validation):
                all_success = False
    elif not args.skip_validation:
        print("⚠️  Pareto-Datei nicht gefunden; überspringe Phase 4")

    # Summary
    print("\n" + "=" * 80)
    if all_success:
        print("✅ THESIS PIPELINE ERFOLGREICH ABGESCHLOSSEN")
    else:
        print("⚠️  THESIS PIPELINE MIT WARNUNGEN ABGESCHLOSSEN")
    print("=" * 80)

    print("\n📊 ERGEBNISSE & PFADE:\n")

    print("1️⃣  EXPLORATION (Phase 1 - Pareto-Front für Thesis-Plots):")
    print(f"   📁 Plots:  {OUTPUT_DIR / 'tuning_weights' / 'pareto'}")
    print(
        f"   📋 CSV:    {OUTPUT_DIR / 'tuning_weights' / 'pareto' / 'pareto_solutions.csv'}"
    )

    print("\n2️⃣  OPTIMIZATION (Phase 2/3 - Best Parameters):")
    optuna_file = find_latest_optuna_file()
    if optuna_file:
        print(f"   📁 Best:   {optuna_file}")
        best_params = load_optuna_best(optuna_file)
        if best_params:
            print("   ✅ Parameters:")
            for key, val in best_params.items():
                print(f"      - {key}: {val}")
    else:
        print("   ⚠️  Optuna-Datei nicht gefunden")

    print("\n3️⃣  VALIDATION (Phase 4 - Robustheit):")
    validation_dir = OUTPUT_DIR / "validation"
    if validation_dir.exists():
        print(f"   📁 Results: {validation_dir}")
        validation_csv = validation_dir / "validation_results.csv"
        if validation_csv.exists():
            print(f"   📋 CSV:     {validation_csv}")
    else:
        print("   ⚠️  Validation Directory nicht gefunden")

    print("\n" + "=" * 80)
    print(f"End: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print("=" * 80 + "\n")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
