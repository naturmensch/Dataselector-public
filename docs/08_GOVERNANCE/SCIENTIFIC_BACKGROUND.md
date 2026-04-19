# Scientific Background & Methodology

> Historical Note (2026-02-23): Diese Seite dokumentiert primär theoretischen
> Hintergrund und ältere methodische Formulierungen. Für den aktiven
> Thesis-Methodikvertrag gelten stattdessen:
> - `../../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md`
> - `../../08_GOVERNANCE/METHODOLOGY.md`
> - `docs/CONFIG_POLICY.md`

**Version:** 1.0  
**Date:** 2. Februar 2026  
**Audience:** Wissenschaftler, Methodenentwickler

## Table of Contents

1. [Overview](#overview)
2. [Facility Location Theory](#facility-location-theory)
3. [Multi-Criteria Optimization](#multi-criteria-optimization)
4. [Spatial Constraints](#spatial-constraints)
5. [Feature Extraction](#feature-extraction)
6. [Bootstrap Uncertainty Quantification](#bootstrap-uncertainty-quantification)
7. [Optuna Hyperparameter Optimization](#optuna-hyperparameter-optimization)
8. [Mathematical Foundations](#mathematical-foundations)

---

## Overview

Dataselector implementiert **submodulare Optimierung** für die Selektion einer diversen Teilmenge historischer Kartenkacheln. Die Methodik kombiniert:

1. **Facility Location Selection** (NP-hard, approximierbar via Greedy)
2. **Multi-Criteria Distance Metric** (Visual + Spatial + Temporal)
3. **Spatial Constraints** (Hard-Constraint für Mindestabstände)
4. **Bayesian Optimization** (Optuna) für Hyperparameter-Tuning
5. **Bootstrap Resampling** für Unsicherheitsquantifizierung

### Problem Statement

**Gegeben:**
- Dataset $\mathcal{D} = \{T_1, T_2, \ldots, T_n\}$ mit $n = 673$ Kartenkacheln
- Feature-Repräsentation $\mathbf{f}_i \in \mathbb{R}^{d}$ für jede Kachel $T_i$ (z.B. $d=768$ für DINOv2)
- Metadaten: Koordinaten $(\text{lat}_i, \text{lon}_i)$, Zeitstempel $\text{year}_i$

**Gesucht:**
- Teilmenge $\mathcal{S} \subset \mathcal{D}$ mit $|\mathcal{S}| = k$ (z.B. $k=50$)
- **Zielfunktion:** Maximiere Diversität unter Einhaltung räumlicher Constraints

$$
\mathcal{S}^* = \arg\max_{\mathcal{S} \subset \mathcal{D}, |\mathcal{S}|=k} f_{\text{diversity}}(\mathcal{S}) \quad \text{s.t.} \quad d_{\text{spatial}}(T_i, T_j) \geq d_{\min} \quad \forall T_i, T_j \in \mathcal{S}
$$

---

## Facility Location Theory

### Submodular Functions

Eine Funktion $f: 2^{\mathcal{D}} \to \mathbb{R}$ ist **submodular**, falls:

$$
f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B) \quad \forall A \subseteq B \subseteq \mathcal{D}, x \notin B
$$

**Intuition:** Diminishing Returns – Der marginale Gewinn eines Elements nimmt ab, wenn die Menge wächst.

### Facility Location Function

**Definition:**

$$
f_{\text{FL}}(\mathcal{S}) = \sum_{i=1}^{n} \max_{j \in \mathcal{S}} \text{sim}(T_i, T_j)
$$

- $\text{sim}(T_i, T_j)$: Ähnlichkeit zwischen Kacheln (z.B. negativer Euklidischer Abstand)
- Interpretation: Jede Kachel trägt zur Gesamtdiversität bei, wie gut sie durch die Selektion "repräsentiert" wird

**Eigenschaften:**
- ✅ Submodular
- ✅ Monoton (hinzufügen von Elementen erhöht nie $f$)
- ✅ Approximierbar via Greedy: $(1 - 1/e)$-optimal (~63%)

### Greedy Algorithm

```python
def greedy_facility_location(D: set, k: int) -> set:
    S = set()
    for _ in range(k):
        # Finde Element mit größtem marginalem Gewinn
        x_best = arg max_{x ∈ D \ S} [f_FL(S ∪ {x}) - f_FL(S)]
        S = S ∪ {x_best}
    return S
```

**Komplexität:** $O(n \cdot k)$ Evaluationen (mit Lazy Greedy: oft deutlich schneller)

---

## Multi-Criteria Optimization

### Unified Distance Metric

**Problem:** Naïve Feature-Augmentation (konkateniere visual, spatial, temporal) führt zu Dominanz hochdimensionaler Features.

**Lösung:** Explizite gewichtete Kombination normalisierter Distanzen:

$$
D_{\text{multi}}(i,j) = \alpha \cdot D_{\text{visual}}^{\text{norm}}(i,j) + \beta \cdot D_{\text{spatial}}^{\text{norm}}(i,j) + \gamma \cdot D_{\text{temporal}}^{\text{norm}}(i,j)
$$

**Constraints:**
- $\alpha + \beta + \gamma = 1$ (Normalisierung)
- $\alpha, \beta, \gamma \in [0, 1]$

### Normalisierung

**Min-Max-Skalierung:**

$$
D^{\text{norm}}(i,j) = \frac{D(i,j) - \min_{i,j} D(i,j)}{\max_{i,j} D(i,j) - \min_{i,j} D(i,j)}
$$

**Eigenschaften:**
- $D^{\text{norm}} \in [0, 1]$
- Verhindert Dominanz einzelner Dimensionen

### Component Distances

#### 1. Visual Distance

Euklidische Distanz im Feature-Space:

$$
D_{\text{visual}}(i,j) = \|\mathbf{f}_i - \mathbf{f}_j\|_2 = \sqrt{\sum_{d=1}^{D} (f_{i,d} - f_{j,d})^2}
$$

- Feature-Dimensionalität: $D = 768$ (DINOv2) oder $D = 2048$ (ResNet50)

**Alternative:** Cosine-Distanz:

$$
D_{\text{cosine}}(i,j) = 1 - \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|}
$$

#### 2. Spatial Distance (Haversine)

Geodätische Distanz auf Erdoberfläche:

$$
d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
$$

- $R = 6371$ km (Erdradius)
- $\phi$: Latitude (Breitengrad)
- $\lambda$: Longitude (Längengrad)

**Implementierung:**

```python
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c
```

**Projected Coordinates (Fallback):**

Falls Koordinaten in projizierten Systemen (EPSG:25832):

$$
D_{\text{spatial}}(i,j) = \sqrt{(\text{east}_i - \text{east}_j)^2 + (\text{north}_i - \text{north}_j)^2} \cdot 10^{-3}
$$

(Division durch 1000 für Meter → Kilometer)

#### 3. Temporal Distance

Absolute Differenz in Jahren:

$$
D_{\text{temporal}}(i,j) = |\text{year}_i - \text{year}_j|
$$

**2D-Replication:** Für konsistente Matrix-Dimensionen:

```python
# Temporal ist 1D (ein Wert pro Kachel)
temporal_1d = np.array([year_1, year_2, ..., year_n])

# Repliziere zu 2D für Distanzberechnung
temporal_2d = np.tile(temporal_1d[:, None], (1, 2))

# Compute pairwise distances
D_temporal = euclidean_distances(temporal_2d, temporal_2d)
```

---

## Spatial Constraints

### Hard Constraint

**Definition:**

$$
\forall T_i, T_j \in \mathcal{S}, i \neq j: \quad d_{\text{haversine}}(\text{lat}_i, \text{lon}_i, \text{lat}_j, \text{lon}_j) \geq d_{\min}
$$

- $d_{\min}$: Mindestabstand (typisch: 40 km)

### Constraint-Integrated Selection

**Greedy mit Constraint-Check:**

```python
def select_with_constraint(D, k, d_min, coords):
    S = []
    for _ in range(k):
        candidates = D - set(S)
        for x in candidates:
            # Check constraint
            valid = True
            for y in S:
                if haversine_distance(coords[x], coords[y]) < d_min:
                    valid = False
                    break
            if valid:
                S.append(x)
                break
    return S
```

**Problem:** Kann fehlschlagen, wenn $d_{\min}$ zu groß.

### Adaptive Relaxation

**Strategie:** Reduziere $d_{\min}$ iterativ um 10%:

$$
d_{\min}^{(t+1)} = 0.9 \cdot d_{\min}^{(t)}
$$

**Algorithmus:**

```python
max_relaxations = 5
for attempt in range(max_relaxations):
    try:
        S = select_with_constraint(D, k, d_min, coords)
        return S
    except InsufficientSamplesError:
        d_min *= 0.9
        print(f"Relaxing to {d_min:.1f} km")
```

### Constraint vs. Multi-Criteria

| Ansatz | Spatial Handling | Trade-Off Control | Pros | Cons |
|--------|------------------|-------------------|------|------|
| **Constraint-Integrated** | Hard constraint | Spatial dominiert | Garantiert $d_{\min}$ | Visual/Temporal untergewichtet |
| **Multi-Criteria** | Via $\beta$ Gewicht | Explizit balancierbar | Vollständige Kontrolle | $d_{\min}$ nicht garantiert |

**Empfehlung:** Multi-Criteria + Post-Processing Constraint-Check

---

## Feature Extraction

### Deep Learning Models

#### ResNet50

**Architecture:**

- **Layers:** 50 (Conv + Residual Blocks)
- **Input:** $224 \times 224$ RGB
- **Output:** $2048$-dim Feature-Vektor (vor Final-FC)
- **Pre-Training:** ImageNet1K (1.28M images)

**Feature-Extraktion:**

```python
import torch
from torchvision import models

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC
model.eval()

with torch.no_grad():
    features = model(images)  # [batch, 2048, 1, 1]
    features = features.squeeze()  # [batch, 2048]
```

**Preprocessing:**

1. Grayscale-Konversion (entfernt Farbrauschen)
2. Autokontrast (verstärkt verblasste Linien)
3. RGB-Rückkonversion (Modellkompatibilität)

#### DINOv2 (Vision Transformer)

**Architecture:**

- **Type:** ViT-Small (22M params)
- **Input:** $384 \times 384$ RGB (flexibel skalierbar)
- **Output:** $768$-dim CLS-Token-Embedding
- **Pre-Training:** Self-supervised (142M images)

**Self-Supervised Learning:**

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^+) / \tau)}{\sum_{k=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

- $\mathbf{z}_i$: Embedding von Bild $i$
- $\mathbf{z}_j^+$: Augmentierte Version von $i$ (positive)
- $\mathbf{z}_k$: Andere Bilder (negatives)

**Vorteile:**
- ✅ Keine Labels erforderlich
- ✅ Generalisiert besser zu Out-of-Distribution-Daten
- ✅ Robuste Repräsentationen für historische Karten

**Feature-Extraktion:**

```python
import torch

# Load DINOv2 model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

with torch.no_grad():
    features = model(images)  # [batch, 768]
```

---

## Bootstrap Uncertainty Quantification

### Methodology

**Bootstrap-Resampling:**

1. Ziehe $n$ Samples mit Replacement aus $\mathcal{D}$: $\mathcal{D}^* = \{T_{i_1}^*, T_{i_2}^*, \ldots, T_{i_n}^*\}$
2. Feature-Extraktion (mit Caching für identische $T_i^*$)
3. Diversity-Selektion mit festen Hyperparametern
4. Metriken-Berechnung
5. Wiederhole $B = 200$ mal

### Jaccard-Überlapp

**Definition:**

$$
J(\mathcal{S}_{\text{orig}}, \mathcal{S}_{\text{boot}}) = \frac{|\mathcal{S}_{\text{orig}} \cap \mathcal{S}_{\text{boot}}|}{|\mathcal{S}_{\text{orig}} \cup \mathcal{S}_{\text{boot}}|}
$$

- $J \in [0, 1]$
- $J = 1$: Perfekte Übereinstimmung
- $J = 0$: Keine Überschneidung

**Interpretation:**
- $J > 0.7$: Hohe Stabilität
- $0.5 < J < 0.7$: Moderate Stabilität
- $J < 0.5$: Niedrige Stabilität (Selektion stark von Daten abhängig)

### Confidence Intervals

**Percentile-Methode:**

$$
\text{CI}_{95\%} = \left[ \text{Percentile}_{2.5}(\{\theta_b\}_{b=1}^B), \text{Percentile}_{97.5}(\{\theta_b\}_{b=1}^B) \right]
$$

- $\theta_b$: Metrik-Wert aus Bootstrap-Iteration $b$

**Beispiel:**

```python
import numpy as np

# 200 Bootstrap-Werte
spatial_mean_dists = [412.3, 415.8, 410.2, ..., 418.5]

# 95% CI
ci_lower = np.percentile(spatial_mean_dists, 2.5)
ci_upper = np.percentile(spatial_mean_dists, 97.5)

print(f"Mean: {np.mean(spatial_mean_dists):.1f} km")
print(f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
```

### Statistical Summary

**Computed Statistics:**

| Statistic | Formula |
|-----------|---------|
| Mean | $\bar{\theta} = \frac{1}{B} \sum_{b=1}^{B} \theta_b$ |
| Std Dev | $\sigma = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} (\theta_b - \bar{\theta})^2}$ |
| 95% CI | $[\text{Percentile}_{2.5}, \text{Percentile}_{97.5}]$ |
| Min/Max | $\min_b \theta_b, \max_b \theta_b$ |

---

## Optuna Hyperparameter Optimization

### Bayesian Optimization

**Ziel:** Finde optimale Hyperparameter $(n, \alpha, \beta, \gamma, d_{\min})$:

$$
\theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}[\mathcal{L}(\theta)]
$$

**Samplers:**

| Sampler | Type | Pros | Cons |
|---------|------|------|------|
| **TPE** (Tree-structured Parzen Estimator) | Bayesian | Schnell, robust | Limitiert für hochdimensionale Spaces |
| **GP** (Gaussian Process) | Bayesian | Sample-effizient | Langsam bei vielen Trials |
| **CMA-ES** (Covariance Matrix Adaptation) | Evolutionär | Robust für kontinuierliche Spaces | Benötigt mehr Trials |

### TPE Algorithm

**Surrogate Model:**

$$
p(\theta | y) \propto p(y | \theta) \cdot p(\theta)
$$

- Modelliert $p(y | \theta)$ mit zwei Kernel-Density-Estimators:
  - $\ell(\theta)$: Density der "guten" Trials ($y < y^*$)
  - $g(\theta)$: Density der "schlechten" Trials ($y \geq y^*$)

**Acquisition Function:**

$$
\text{EI}(\theta) = \frac{\ell(\theta)}{g(\theta)}
$$

**Nächster Trial:**

$$
\theta_{\text{next}} = \arg\max_{\theta} \text{EI}(\theta)
$$

### CMA-ES Algorithm

**Update-Regel:**

$$
\mathbf{m}^{(t+1)} = \mathbf{m}^{(t)} + \eta \cdot \mathbf{C}^{(t)} \mathbf{z}^{(t)}
$$

- $\mathbf{m}$: Mean-Vektor
- $\mathbf{C}$: Kovarianzmatrix
- $\mathbf{z}$: Gradientenschätzung

**Adaptiv:** Passt Kovarianzmatrix basierend auf Erfolgshistorie an.

### Optuna Trial Workflow

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    n_samples = trial.suggest_int("n_samples", 20, 100)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0)
    gamma = 1.0 - alpha - beta  # Constraint
    
    # Run pipeline
    metrics = run_pipeline(n_samples, alpha, beta, gamma)
    
    # Composite objective (minimize)
    return -metrics["spatial_mean_dist"] + metrics["temporal_std"] * 0.1

# Create study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.CmaEsSampler(seed=42)
)

# Optimize
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

---

## Mathematical Foundations

### Submodularity Proof (Facility Location)

**Theorem:** $f_{\text{FL}}(\mathcal{S}) = \sum_{i=1}^{n} \max_{j \in \mathcal{S}} \text{sim}(T_i, T_j)$ ist submodular.

**Proof:**

Seien $A \subseteq B \subseteq \mathcal{D}$ und $x \notin B$. Zeige:

$$
f_{\text{FL}}(A \cup \{x\}) - f_{\text{FL}}(A) \geq f_{\text{FL}}(B \cup \{x\}) - f_{\text{FL}}(B)
$$

Für jedes $T_i$:

$$
\Delta_A(T_i) = \max\{\max_{j \in A} \text{sim}(T_i, T_j), \text{sim}(T_i, x)\} - \max_{j \in A} \text{sim}(T_i, T_j)
$$

$$
\Delta_B(T_i) = \max\{\max_{j \in B} \text{sim}(T_i, T_j), \text{sim}(T_i, x)\} - \max_{j \in B} \text{sim}(T_i, T_j)
$$

Da $A \subseteq B$: $\max_{j \in A} \text{sim}(T_i, T_j) \leq \max_{j \in B} \text{sim}(T_i, T_j)$

$\Rightarrow \Delta_A(T_i) \geq \Delta_B(T_i)$ (Diminishing Returns)

$\Rightarrow f_{\text{FL}}(A \cup \{x\}) - f_{\text{FL}}(A) = \sum_{i} \Delta_A(T_i) \geq \sum_{i} \Delta_B(T_i) = f_{\text{FL}}(B \cup \{x\}) - f_{\text{FL}}(B)$ ∎

### Greedy Approximation Guarantee

**Theorem (Nemhauser et al., 1978):** Für monotone submodulare Funktionen garantiert Greedy:

$$
f(\mathcal{S}_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) \cdot f(\mathcal{S}_{\text{optimal}})
$$

**Approximation Factor:** $\approx 0.632$ (63.2% des Optimums)

**Proof Sketch:**

Sei $\mathcal{O} = \{o_1, o_2, \ldots, o_k\}$ die optimale Lösung. Nach $t$ Greedy-Iterationen:

$$
f(\mathcal{O}) - f(\mathcal{S}_t) \leq k \cdot \max_{x \in \mathcal{O}} [f(\mathcal{S}_t \cup \{x\}) - f(\mathcal{S}_t)]
$$

Da Greedy das beste $x$ wählt:

$$
f(\mathcal{S}_{t+1}) - f(\mathcal{S}_t) \geq \frac{f(\mathcal{O}) - f(\mathcal{S}_t)}{k}
$$

Telescope über $t = 0, \ldots, k-1$:

$$
f(\mathcal{S}_k) \geq \left(1 - \left(1 - \frac{1}{k}\right)^k\right) \cdot f(\mathcal{O}) \geq \left(1 - \frac{1}{e}\right) \cdot f(\mathcal{O})
$$

### Haversine Formula Derivation

**Ausgangspunkt:** Sphärisches Gesetz des Cosinus:

$$
\cos(c) = \cos(a) \cos(b) + \sin(a) \sin(b) \cos(C)
$$

**Anwendung auf Erde:**

- $a = \frac{\pi}{2} - \phi_1$, $b = \frac{\pi}{2} - \phi_2$, $C = \Delta\lambda$

$$
\cos(c) = \sin(\phi_1) \sin(\phi_2) + \cos(\phi_1) \cos(\phi_2) \cos(\Delta\lambda)
$$

**Haversine-Transformation:** $\text{hav}(\theta) = \sin^2(\theta/2)$

$$
\text{hav}(c) = \text{hav}(\Delta\phi) + \cos(\phi_1) \cos(\phi_2) \cdot \text{hav}(\Delta\lambda)
$$

$$
c = 2 \arcsin(\sqrt{\text{hav}(c)})
$$

$$
d = R \cdot c
$$

---

## Related Documentation

- [Architecture](../02_THEORY/architecture.md) – System Design
- [API Reference](../06_REFERENCE/api_reference.md) – Programming Interface
- [Thesis Pipeline How-To](../03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md) – Canonical workflow
- [07_ARCHIVE/legacy_xxl_ops/](../07_ARCHIVE/legacy_xxl_ops/) – Historical XXL / monitoring docs

---

**References:**

1. Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *An analysis of approximations for maximizing submodular set functions—I.* Mathematical programming, 14(1), 265-294.
2. Wei, K., Iyer, R., & Bilmes, J. (2015). *Submodularity in data subset selection and active learning.* ICML.
3. Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.

**Maintainer:** Sebastian (seb@dataselector.de)  
**License:** MIT
