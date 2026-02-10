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

### 2. Python-Umgebung einrichten (kanonisch)

```bash
# Kanonische Runtime: micromamba run
micromamba create -n dataselector -f environment.yml -y
micromamba run -n dataselector python -V

# Kompatibilitäts-Wrapper (optional, delegiert auf micromamba/conda)
./scripts/exec_in_env.sh --env dataselector --create --yes -- python -V
```

### 3. Dependencies installieren

```bash
micromamba run -n dataselector pip install -r requirements-cpu.txt
```

## Projektstruktur

```
Dataselector/
├── dataselector/                 # Hauptmodule (kanonische API)
│   ├── cli.py                    # Unified CLI
│   ├── data/                     # Datenzugriff / Build-Tools
│   ├── features/                 # Deep Learning Features
│   ├── selection/                # UMAP/K-Means + Selection
│   ├── pipeline/                 # ExperimentManager / Pipeline helpers
│   └── workflows/                # Kanonische Workflows
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

> Thesis-Runbook (authoritative): `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
>
> Diese Datei ist die zentrale Quelle fuer Setup, Gates, deterministische
> Twin-Run-Pruefung und Go/No-Go vor Annotation.
>
> **Kanonische Ausfuehrung:** `micromamba run -n dataselector <command>`
>
> Hinweis: Falls unten Beispiele ohne Prefix stehen, fuehre sie im aktiven
> `dataselector`-Env aus oder setze den Prefix explizit davor.

### Daten vorbereiten

1. Platzieren Sie die Metadaten-Datei (`KDR100_foliage_with_files_epsg3857.csv` oder `all_png_tiles.dbf`) im `data/` Verzeichnis

   Hinweis: Falls die Rohbilder und Sidecar-XMLs in `data/images/` vorhanden sind, kann `python -m dataselector build-tiles` automatisch `data/new_all_tiles.csv` erzeugen (z.B. `python -m dataselector build-tiles --image-dir data/images --out data/new_all_tiles.csv`).
2. Erstellen Sie einen Ordner `data/images/` und legen Sie die Kartenbilder dort ab

### Vollständiger Experiment-Workflow (Modern: Autoscale → Sampler Suite → XXL)

Der vollständige, wissenschaftlich fundierte Ablauf besteht aus drei klaren Phasen:

1. **Autoscale** (`python -m dataselector autoscale`) — gestufte Suche nach sinnvoller `n_samples` und globalen Hyperparametern (Stages z.B. 50 → 100 → 300 → full). Ergebnis: `outputs/optuna_autoscale_selected_n_samples.txt` und `outputs/optuna_autoscale_best_latest.json`.

2. **Sampler Suite** (`python -m dataselector thesis-sampler-suite`) — Vergleicht Sampler (QMC, TPE, CMA‑ES) über mehrere Seeds und verwendet die Autoscale‑Ergebnisse zur Einschränkung der Suchräume (Constrained Bounds). Ergebnis: `outputs/selected_sampler.json` und per‑run `results/`-Ordner.

3. **XXL Pipeline (Phases 0–5)** (`python -m dataselector xxl`) — Validierung (Phase 0), große Optimierungsläufe (Phase 1–4), Bootstrap UQ (Phase 5) und Erstellung der Thesis‑Artefakte.

Für die komplette Ausführung nutze die CLI‑Abfolge (Autoscale in der Suite aktiviert):

```bash
# Vollständige Orchestrierung (Autoscale → Sampler Suite → XXL)
python -m dataselector thesis-sampler-suite --autoscale
python -m dataselector xxl
```

Wenn Sie nur die Sampler-Suite mit Autoscale ausführen möchten:

```bash
python -m dataselector thesis-sampler-suite --autoscale
```

Und falls Sie die Suite ohne Autoscale durchführen wollen:

```bash
python -m dataselector thesis-sampler-suite
```

Nur die moderne XXL‑Orchestration (z.B. nach erfolgreicher Suite) läuft so:

```bash
python -m dataselector xxl --best-sampler tpe
```

Hinweis: Die Orchestrator‑Skripte prüfen die Existenz von Artefakten im `outputs/`-Verzeichnis (`optuna_autoscale_*`, `selected_sampler.json`) und verwenden diese automatisiert. Die Skripte brauchen eine Umgebung mit `optuna` installiert, wenn Optuna‑Phasen ausgeführt werden.

Provenance & Reproduzierbarkeit:
- Das Orchestrator-Skript kopiert sämtliche relevanten Artefakte in `outputs/runs/run_<TIMESTAMP>/`, darunter die `optuna_results.csv`, ggf. die `optuna_study.pkl`, eine `pipeline_config.optuna.yaml` (oder die Backup-Datei bei Injection) und die finalen CSV/Plots. So sind alle Eingaben dokumentiert.

Schneller Smoke-Run (lokal / CI):

```bash
# Schneller Test: kleiner Optuna Run (2 Trials) und Unit-Tests
pytest -q
python -m dataselector optuna-optimize --n-trials 2 --n-candidates 50 --dim 32 --n-samples 5 --min-distance-km 10
```

Diese Commands sind absichtlich klein gehalten, damit sie schnell laufen und als Smoke-Test in CI nutzbar sind.

## Administrative Tools

For workspace management and validation, use the canonical CLI commands:

```bash
# Validate GIS dependencies
python -m dataselector check-geo

# Verify protected file paths (prevent accidental commits)
python -m dataselector check-protected --list

# Audit workspace integrity
python -m dataselector check-env

# Check and fix documentation links
python -m dataselector docs-link-check
python -m dataselector docs-link-autofix --yes

# Clean up temporary artifacts and caches
python -m dataselector clean-workspace
python -m dataselector clean-workspace --delete-outputs --delete-cache --delete-venvs --yes

# Validate CSV vs Raster alignment
python -m dataselector align-audit --csv data/new_all_tiles.csv --base-dir data/images

# List archived experiment outputs
python -m dataselector list-archives
```

For detailed documentation on all tools, see [Administrative Tools Reference](docs/06_REFERENCE/TOOLS_REFERENCE.md).

### Pipeline ausführen (modern)

Die empfohlene Methode ist die 3‑Phasen Orchestrierung (Autoscale → Sampler Suite → XXL). Für den kompletten Durchlauf benutze:

```bash
python -m dataselector thesis-sampler-suite --autoscale
python -m dataselector xxl
```

Alternativen für gezielte Ausführung einzelner Schritte:

- Nur Autoscale (Schneller Test / Debugging):
```bash
python -m dataselector autoscale --n-trials 20 --stages 50 100 --n-candidates 100
```

- Sampler Suite (mit oder ohne Autoscale):
```bash
# Mit automatischem Autoscale
python -m dataselector thesis-sampler-suite --autoscale

# Ohne Autoscale
python -m dataselector thesis-sampler-suite
```

- Nur XXL Pipeline (nach Suite):
```bash
python -m dataselector xxl --best-sampler tpe
```

Die allgemeinen Pipeline-Schritte (Metadaten → Feature Extraction → Clustering → Selection → Visualisierung) bleiben als konzeptionelles Gerüst erhalten; die Orchestrations-Skripte fügen die wissenschaftlichen Optimierungs- und Validierungsphasen hinzu (Autoscale / Sampler‑Suite / XXL).
### Konfiguration anpassen

Bearbeiten Sie `config/pipeline_config.yaml`:

```yaml
# Anzahl auszuwählender Samples ändern
selection:
  n_samples: 24  # aktueller Thesis-Policy-Stand
  
# Clustering-Parameter
clustering:
  n_clusters: 8  # Mehr/weniger Cluster
  
# Räumliche Constraints
selection:
  min_distance_km: 28.5  # Operative Policy; geometrische Referenz siehe Reports
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
from dataselector.data.metadata_processor import MetadataProcessor

processor = MetadataProcessor("data/all_png_tiles.dbf")  # unterstützt auch .dbf
df = processor.load_csv()
df = processor.add_temporal_metadata()  # Extrahiert Jahr aus Dateinamen

# Optional: DBF -> CSV konvertieren
# csv_path = processor.convert_dbf_to_csv('data/converted_metadata.csv')
# print(f"CSV erzeugt: {csv_path}")
```

### 2. Feature Extraction

```python
from dataselector.features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(model_name='resnet50')
features = extractor.extract_features_batch(
    image_paths=df['longName'].tolist(),
    data_dir=Path("data/images")
)
```

### 3. Clustering

```python
from dataselector.selection.clustering import ClusteringPipeline

clustering = ClusteringPipeline(n_clusters=8)
embeddings_2d, labels = clustering.fit_transform(features)
```

### 4. Diversity Selection

```python
from dataselector.selection.diversity_selector import DiversitySelector

selector = DiversitySelector(n_samples=24)
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
- Ein sicherer CLI‑Befehl steht zur Verfügung, um Kandidaten zu listen, zu archivieren oder optional zu löschen:

```bash
# nur prüfen (Dry-Run ist Standard)
python -m dataselector clean-workspace

# Ausgewählte Ordner löschen (vorsichtig!)
python -m dataselector clean-workspace --delete-outputs --delete-venvs --yes

# Archivieren statt löschen
python -m dataselector clean-workspace --delete-outputs --archive /path/to/archive.tar.gz --yes
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
  min_distance_km: 25.0  # Statt 28.5
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
> Der operative Default für `min_distance_km` ist 28.5 km (siehe `config/pipeline_config.yaml`).
> Die geometrische Referenz liegt separat bei ~45.0 km und ist in den Decision-Reports dokumentiert.
> Das Logging gibt explizit an, ob der räumliche Constraint aktiv ist (`min_dist=...km`) oder deaktiviert (`min_dist=0.0km (disabled)`).
