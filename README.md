# KDR100 Datenselektion

**Algorithmische Datenselektion für die Karte des Deutschen Reiches (KDR100) mittels Unsupervised Deep Clustering**

## Überblick

Dieses Projekt implementiert einen hybriden Active-Learning-Workflow zur objektiven Auswahl von Trainingsbeispielen aus dem heterogenen KDR100-Kartendatensatz. Anstatt manueller "Hand-Picking"-Methoden nutzt das System Deep Learning und submodulare Optimierung, um mathematisch optimale, diverse Samples zu identifizieren.

### Kernfunktionalitäten

- **Feature Extraction**: Extraktion visueller Features mittels vortrainiertem ResNet50
- **Dimensionsreduktion**: UMAP-Projektion zur Visualisierung des Feature-Raums
- **Clustering**: K-Means zur automatischen Erkennung von Landschaftstypen
- **Diversity Sampling**: Facility Location Function für maximale Coverage
- **Constraint Handling**: Zeitliche und räumliche Diversitäts-Constraints

## Technischer Stack

- **Python**: 3.9+
- **Deep Learning**: PyTorch, torchvision
- **Machine Learning**: scikit-learn, umap-learn
- **Optimierung**: apricot-select (submodulare Optimierung)
- **Datenverarbeitung**: pandas, numpy
- **Visualisierung**: matplotlib, seaborn

## Installation

### 1. Repository klonen

```bash
git clone <repository-url>
cd Dataselector
```

### 2. Python-Umgebung einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

## Projektstruktur

```
Dataselector/
├── src/                          # Hauptmodule
│   ├── __init__.py
│   ├── main.py                   # Haupt-Pipeline
│   ├── metadata_processor.py    # CSV/DBF-Verarbeitung
│   ├── feature_extractor.py     # Deep Learning Features
│   ├── clustering.py             # UMAP + K-Means
│   ├── diversity_selector.py    # Facility Location
│   └── visualizer.py             # Visualisierungen
├── data/                         # Datenverzeichnis
│   ├── KDR100_foliage_with_files_epsg3857.csv
│   └── images/                   # Kartenbilder
├── config/                       # Konfigurationsdateien
│   └── pipeline_config.yaml
├── notebooks/                    # Jupyter Notebooks
├── outputs/                      # Ergebnisse
├── requirements.txt
└── README.md
```

## Nutzung

### Daten vorbereiten

1. Platzieren Sie die Metadaten-Datei (`KDR100_foliage_with_files_epsg3857.csv` oder `all_png_tiles.dbf`) im `data/` Verzeichnis
2. Erstellen Sie einen Ordner `data/images/` und legen Sie die Kartenbilder dort ab

### Vollständiger Experiment-Workflow (Coarse → Fine → Optuna → Bootstrap → Final)

Für das intensivste, reproduzierbare Lauf-Setup nutzen Sie das mitgelieferte Orchestrator-Skript `scripts/run_full_experiment.sh`.

Beispiel (komplett, interaktiv):

```bash
# Standard Lauf: coarse sweep → fine sweep → Optuna → Bootstrap → final selection
./scripts/run_full_experiment.sh
```

Nicht-interaktiv, Optuna mit 300 Trials und 300 Bootstrap-Resamples:

```bash
./scripts/run_full_experiment.sh --n-trials 300 --n-boot 300 --yes
```

Wichtige Flags:
- `--use-optuna-best` : Extrahiert nach Optuna den besten Trial und schreibt eine konfigurationsdatei in den Experiment-Ordner (`outputs/experiments/run_<TS>/pipeline_config.optuna.yaml`).
- `--inject-optuna` : Injiziert den besten Optuna-Trial direkt in `config/pipeline_config.yaml` (vorher wird ein Backup `config/pipeline_config.yaml.optuna_bak` angelegt).
- `--final-with-optuna-config` : Führt den finalen Run temporär mit der generierten Optuna-Konfiguration aus (Original-Konfig wird danach wiederhergestellt).

Provenance & Reproduzierbarkeit:
- Das Orchestrator-Skript kopiert sämtliche relevanten Artefakte in `outputs/experiments/run_<TIMESTAMP>/`, darunter die `optuna_results.csv`, ggf. die `optuna_study.pkl`, eine `pipeline_config.optuna.yaml` (oder die Backup-Datei bei Injection) und die finalen CSV/Plots. So sind alle Eingaben dokumentiert.

Schneller Smoke-Run (lokal / CI):

```bash
# Schneller Test: kleiner Optuna Run (2 Trials) und Unit-Tests
pytest -q
python scripts/optuna_optimize.py --n-trials 2 --n-candidates 50 --dim 32 --n-samples 5 --min-distance-km 10
```

Diese Commands sind absichtlich klein gehalten, damit sie schnell laufen und als Smoke-Test in CI nutzbar sind.


### Pipeline ausführen

```bash
python src/main.py
```

Die Pipeline führt automatisch folgende Schritte aus:

1. **Metadaten-Verarbeitung**: Lädt CSV und extrahiert temporale/räumliche Informationen
2. **Feature Extraction**: Generiert 2048-dimensionale Vektoren für jede Kachel
3. **Clustering**: Reduziert auf 2D mittels UMAP und gruppiert mit K-Means
4. **Diversity Selection**: Wählt 34 optimale Samples via Facility Location
5. **Visualisierung**: Erstellt Plots und Zusammenfassungen

### Konfiguration anpassen

Bearbeiten Sie `config/pipeline_config.yaml`:

```yaml
# Anzahl auszuwählender Samples ändern
selection:
  n_samples: 34  # Standard: 5% (~34 von 673)
  
# Clustering-Parameter
clustering:
  n_clusters: 8  # Mehr/weniger Cluster
  
# Räumliche Constraints
selection:
  min_distance_km: 50.0  # Minimale Distanz zwischen Samples
```

## Ausgaben

Nach dem Durchlauf finden Sie im `outputs/` Verzeichnis:

### CSV-Dateien
- `kdr100_selection_selected.csv`: Liste der ausgewählten Kacheln mit Metadaten

### Visualisierungen
- `umap_clusters.png`: 2D-Projektion aller Kacheln mit Cluster-Farben
- `temporal_distribution.png`: Histogramm der zeitlichen Verteilung
- `spatial_distribution.png`: Geografische Karte der Auswahl
- `cluster_distribution.png`: Balkendiagramm der Cluster-Coverage

## Workflow-Details

### 1. Metadaten-Extraktion

```python
from src.metadata_processor import MetadataProcessor

processor = MetadataProcessor("data/all_png_tiles.dbf")  # unterstützt auch .dbf
df = processor.load_csv()
df = processor.add_temporal_metadata()  # Extrahiert Jahr aus Dateinamen

# Optional: DBF -> CSV konvertieren
# csv_path = processor.convert_dbf_to_csv('data/converted_metadata.csv')
# print(f"CSV erzeugt: {csv_path}")
```

### 2. Feature Extraction

```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(model_name='resnet50')
features = extractor.extract_features_batch(
    image_paths=df['longName'].tolist(),
    data_dir=Path("data/images")
)
```

### 3. Clustering

```python
from src.clustering import ClusteringPipeline

clustering = ClusteringPipeline(n_clusters=8)
embeddings_2d, labels = clustering.fit_transform(features)
```

### 4. Diversity Selection

```python
from src.diversity_selector import DiversitySelector

selector = DiversitySelector(n_samples=34)
selected_indices = selector.select(
    features,
    metadata=df,
    temporal_weight=0.2
)
```

### Pre-selection / Seeding (optional)

Wenn bereits annotierte Kacheln vorhanden sind (z. B. Hamburg), können diese als *Seed* in den Selektionsprozess integriert werden, sodass der Algorithmus die verbleibenden N-1 Samples optimiert.

Beispiel (Config):
```yaml
selection:
  pre_selected_names: ['Hamburg']  # oder pre_selected_indices: [145]
```

Das ist wissenschaftlich sauber: es handelt sich um eine Randbedingung (conditional selection), keine nachträgliche Manipulation der Ergebnisse.

## Jupyter Notebooks

Für interaktive Exploration stehen Notebooks im `notebooks/` Verzeichnis zur Verfügung:

```bash
jupyter notebook notebooks/
```

Empfohlene Notebooks (zu erstellen):
- `01_data_exploration.ipynb`: CSV-Analyse und Visualisierung
- `02_feature_analysis.ipynb`: Feature-Extraktion testen
- `03_clustering_experiments.ipynb`: Cluster-Parameter optimieren
- `04_selection_validation.ipynb`: Ergebnisse validieren

## Anpassungen und Erweiterungen

### Alternatives Modell verwenden

```python
# In config/pipeline_config.yaml
feature_extraction:
  model: "dinov2"  # Statt resnet50
```

## Aufräumen & freigeben von Speicherplatz 🔧

Große, generierte Artefakte wie `data/images/`, lokale virtuelle Umgebungen (`.venv`, `venv/`) oder `outputs/validation/` sollten nicht im Git-Repository versioniert werden.

- `.gitignore` wurde erweitert, um typische Artefakte auszuschließen.
- Ein sicheres Script steht zur Verfügung, um Kandidaten zu listen, zu archivieren oder optional zu löschen:

```bash
# nur prüfen
python scripts/clean_workspace.py --dry-run

# Ausgewählte Ordner löschen (vorsichtig!)
python scripts/clean_workspace.py --delete-outputs --delete-venvs

# Archivieren
python scripts/clean_workspace.py --archive data/images /path/to/archive.tar.gz
```

Bitte prüfen Sie die Ausgabe des Dry-Runs, bevor Sie etwas löschen. Wenn Dateien bereits in Git committed sind, entfernen Sie diese erst aus dem Repo (z. B. `git rm -r --cached <path>`) und committen Sie die `.gitignore`-Änderung.

### Mehr Cluster

```python
# In config/pipeline_config.yaml
clustering:
  n_clusters: 12  # Mehr granulare Gruppierung
```

### Zeitliche Gewichtung erhöhen

```python
# In config/pipeline_config.yaml
selection:
  temporal_weight: 0.5  # Höheres Gewicht auf zeitliche Diversität
```

## Wissenschaftliche Grundlagen

### Facility Location Function

Die submodulare Optimierung maximiert folgende Zielfunktion:

$$
F(S) = \sum_{i \in V} \max_{j \in S} \text{sim}(i, j)
$$

wobei $S$ die Auswahl, $V$ alle Samples und $\text{sim}$ eine Ähnlichkeitsfunktion ist.

### UMAP

Uniform Manifold Approximation and Projection reduziert hochdimensionale Features unter Erhalt der topologischen Struktur:

- **Metrik**: Cosine-Similarity (optimal für visuelle Features)
- **n_neighbors**: Kontrolliert lokale vs. globale Struktur
- **min_dist**: Minimale Distanz zwischen Punkten in 2D

## Troubleshooting

### CUDA nicht verfügbar

```python
# In config/pipeline_config.yaml
feature_extraction:
  device: "cpu"  # Erzwingt CPU-Nutzung
```

### Out of Memory bei Feature Extraction

```python
# Reduziere Batch Size
feature_extraction:
  batch_size: 4  # Statt 8
```

### Zu wenige Samples passieren räumlichen Filter

```python
# Verringere Mindestdistanz
selection:
  min_distance_km: 25.0  # Statt 50.0
```

## Entwicklungsrichtlinien

- **Type Hints**: Alle Funktionen verwenden Type Annotations
- **Docstrings**: Google-Style Dokumentation
- **Modularität**: Jede Komponente als eigenständiges Modul
- **Konfigurierbarkeit**: Parameter in YAML auslagern

## Lizenz

[Lizenz hier einfügen]

## Kontakt

[Kontaktinformationen]

## Referenzen

- ResNet: He et al., "Deep Residual Learning for Image Recognition" (2016)
- UMAP: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018)
- Submodular Optimization: Krause & Golovin, "Submodular Function Maximization" (2014)

> **Hinweis:**
> Der empfohlene Default für `min_distance_km` ist 50.0 km (siehe `config/pipeline_config.yaml`).
> In Experimenten und Skripten sollte dieser Wert übernommen werden, um räumliche Diversität zu gewährleisten.
> Das Logging gibt explizit an, ob der räumliche Constraint aktiv ist (`min_dist=...km`) oder deaktiviert (`min_dist=0.0km (disabled)`).
