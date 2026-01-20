#!/usr/bin/env python3
"""
Diagnose-Skript für KDR100 Datenselektion.
Prüft:
1. Geodätische Validität (Grad vs Meter)
2. Software-Versionen (cmaes, optuna)
3. Legacy-Altlasten in Skripten
"""
import sys
import importlib.metadata
import pandas as pd
from pathlib import Path

def check_geodesy():
    print("\n🔍 1. GEODÄSIE-CHECK")
    print("---------------------")
    
    # Suche Metadata
    candidates = [
        Path("outputs/metadata.csv"),
        Path("data/new_all_tiles.csv"),
        Path("data/all_png_tiles_final_ultimative.csv")
    ]
    
    df = None
    found_path = None
    
    for p in candidates:
        if p.exists():
            print(f"Lade Metadaten von: {p}")
            try:
                df = pd.read_csv(p)
                found_path = p
                break
            except Exception as e:
                print(f"  Fehler beim Lesen von {p}: {e}")
    
    if df is None:
        print("❌ Keine Metadaten-Datei gefunden. Kann Koordinaten nicht prüfen.")
        return

    # Prüfe Spalten
    if 'N' not in df.columns or 'left' not in df.columns:
        print(f"❌ Spalten 'N' und 'left' fehlen in {found_path}.")
        print(f"   Vorhandene Spalten: {list(df.columns)}")
        return

    # Prüfe Wertebereiche
    n_min, n_max = df['N'].min(), df['N'].max()
    left_min, left_max = df['left'].min(), df['left'].max()
    
    print(f"  Wertebereich 'N' (Lat?):   {n_min:.4f} ... {n_max:.4f}")
    print(f"  Wertebereich 'left' (Lon?): {left_min:.4f} ... {left_max:.4f}")
    
    # Heuristik: Deutschland liegt ca. bei Lat 47-55, Lon 6-15.
    # Meter-Koordinaten (GK/UTM) wären > 1.000.000.
    
    is_latlon = (-90 <= n_min and n_max <= 90) and (-180 <= left_min and left_max <= 180)
    
    if is_latlon:
        print("✅ ERGEBNIS: Koordinaten sind wahrscheinlich DEZIMALGRAD (WGS84).")
        print("   -> Die Nutzung der Haversine-Formel ist KORREKT.")
    else:
        print("⚠️  WARNUNG: Koordinaten liegen außerhalb des Grad-Bereichs!")
        print("   -> Es handelt sich wahrscheinlich um PROJIZIERTE METER-KOORDINATEN.")
        print("   -> Die Nutzung der Haversine-Formel ist FALSCH.")
        print("   -> ACTION REQUIRED: Reprojektion oder Umstellung auf Euklidische Distanz.")

def check_versions():
    print("\n🔍 2. VERSION-CHECK")
    print("-------------------")
    
    # Optuna
    try:
        import optuna
        print(f"✅ Optuna: {optuna.__version__}")
    except ImportError:
        print("❌ Optuna: NICHT INSTALLIERT")
        
    # cmaes
    try:
        ver = importlib.metadata.version("cmaes")
        print(f"✅ cmaes:  {ver}")
    except importlib.metadata.PackageNotFoundError:
        print("⚠️  cmaes:  NICHT INSTALLIERT (Kritisch für CMA-ES Sampler!)")

if __name__ == "__main__":
    check_geodesy()
    check_versions()