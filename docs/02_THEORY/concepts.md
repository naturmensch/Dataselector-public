# Concepts & Algorithms

**Version:** 1.0  
**Date:** 2. Februar 2026  
**Status:** Production Ready

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Multi-Criteria Facility Location Algorithm](#multi-criteria-facility-location-algorithm)
3. [Key Algorithms](#key-algorithms)
4. [Constraint Handling](#constraint-handling)

---

## Core Concepts

### Diversity Metrics (α, β, γ)

**Definition:** Weighting factors for visual, spatial and temporal diversity

- **α** – Visual diversity weight ($\alpha \in [0, 1]$, default: 0.40)
- **β** – Spatial diversity weight ($\beta \in [0, 1]$, default: 0.30)
- **γ** – Temporal diversity weight ($\gamma \in [0, 1]$, default: 0.30)

**Constraint:** $\alpha + \beta + \gamma = 1.0$ (normalized to 1.0)

**Interpretation:**
- Higher $\alpha$ → Prioritize visual diversity (more varied imagery)
- Higher $\beta$ → Prioritize spatial distribution (geographic spread)
- Higher $\gamma$ → Prioritize temporal variation (different time periods)

### Spatial Constraint (d_min)

**Definition:** Minimum distance constraint between selected tiles

- **d_min** – Minimum distance in kilometers (default: 40 km)
- Hard constraint: No two selected tiles can be closer than d_min

**Purpose:** Ensure geographic diversity & avoid spatial clustering

**Implementation:** Haversine distance formula for geodetic distance

### Facility Location Selection

**Definition:** Submodular optimization objective

- **Principle:** Maximize coverage of the parameter space
- **Implementation:** Greedy algorithm via `apricot-select` library
- **Approximation:** $(1 - 1/e)$-optimal (~63% of theoretical maximum)

**Submodular Property:** Diminishing returns – adding more samples to a diverse set yields fewer additional benefits than to a homogeneous set

### Feature Representation

**UMAP** – Dimensionality reduction for visualization & clustering
- Reduces high-dimensional features (e.g., 2048D) to 2D
- Preserves local + global structure better than PCA
- Parameters: `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`

**DINOv2 / ResNet50** – Feature extractors used for visual embeddings
- DINOv2 (Vision Transformer): State-of-the-art, self-supervised
- ResNet50 (ImageNet): Faster, more established, robust
- Output: 768D (DINOv2) or 2048D (ResNet50) feature vectors

### Bootstrap Uncertainty Quantification

**Definition:** Resampling method for uncertainty quantification

- **Principle:** Estimate confidence intervals without parametric assumptions
- **Method:** Resample with replacement from original dataset, recompute metrics
- **Standard:** 200 bootstrap samples for 95% confidence intervals
- **Output:** Mean ± 1.96 × SE (95% CI)

---

## Multi-Criteria Facility Location Algorithm

### Problem Formulation

**Maximize:**

$$
f_{\text{diversity}}(\mathcal{S}) = \sum_{i=1}^{n} \max_{j \in \mathcal{S}} D_{\text{multi}}(i, j)
$$

where the **combined distance metric** is:

$$
D_{\text{multi}}(i,j) = \alpha \cdot D_{\text{visual}}^{\text{norm}}(i,j) + \beta \cdot D_{\text{spatial}}^{\text{norm}}(i,j) + \gamma \cdot D_{\text{temporal}}^{\text{norm}}(i,j)
$$

**Subject to:**
- $|\mathcal{S}| = k$ (select exactly k samples)
- $d_{\text{spatial}}(T_i, T_j) \geq d_{\min}$ ∀ $T_i, T_j \in \mathcal{S}$ (spatial constraint)

### Distance Components

#### Visual Distance
Euclidean distance in feature space (after feature extraction):

$$
D_{\text{visual}}(i,j) = \|\mathbf{f}_i - \mathbf{f}_j\|_2
$$

**Normalization:**
```python
D_visual_norm = (D_visual - D_visual.min()) / (D_visual.max() - D_visual.min())
```

#### Spatial Distance
Haversine geodetic distance on Earth surface:

$$
d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
$$

where $R = 6371$ km, $\phi$ = latitude, $\lambda$ = longitude

#### Temporal Distance
Year difference between tiles:

$$
D_{\text{temporal}}(i,j) = |\text{year}_i - \text{year}_j|
$$

**2D Replication:** Temporal is 1D, replicated to 2D for matrix operations

### Normalization Strategy

Each distance component is independently normalized to [0, 1]:

```python
def normalize_distances(D):
    return (D - D.min()) / (D.max() - D.min())

D_combined = alpha * normalize_distances(D_visual) + \
             beta * normalize_distances(D_spatial) + \
             gamma * normalize_distances(D_temporal)
```

**Purpose:** Prevent high-dimensional features from dominating low-dimensional ones

---

## Key Algorithms

### Greedy Facility Location

**Algorithm (Lazy Greedy variant):**

```
Input: Distance matrix D, number of samples k
Output: Selected indices S

S ← {}
gains ← precompute_gains(D)  // O(n²)

for i = 1 to k:
    x_best ← argmax_{x ∉ S} marginal_gain(x, S, gains)
    S ← S ∪ {x_best}
    update_gains(gains, x_best)  // Incremental update

return S
```

**Complexity:** $O(n \cdot k)$ with lazy evaluation (often $\ll n \cdot k$ in practice)

**Approximation:** $(1 - 1/e) \approx 0.63$ of optimal solution

### Adaptive Distance Relaxation

When spatial constraints are unsatisfiable (no feasible solution exists):

```
tolerance ← d_min

while not feasible:
    tolerance ← tolerance × 0.9  # Reduce by 10%
    S ← select_with_constraint(tolerance)
    
    if feasible or tolerance < 1.0:
        break

if not feasible:
    warn("Could not satisfy spatial constraints")
    return S with relaxed constraints
```

**Fallback:** Return samples with best achievable diversity

---

## Constraint Handling

### Hard Spatial Constraint

**During Selection:**

```python
for candidate in candidates:
    distances_to_selected = haversine_distances(candidate, selected)
    
    if all(distances_to_selected >= min_distance_km):
        select(candidate)
```

**Ensures:** Every selected tile is at least `min_distance_km` away from others

### Soft Constraint Relaxation

If hard constraint cannot be satisfied with k samples:

1. **Stage 1:** Try with d_min as-is
2. **Stage 2:** Relax d_min by 10%
3. **Stage 3:** Relax d_min by 20%
4. **Stage 4+:** Continue relaxing until feasible or min_distance = 0

**Logging:** Report final effective distance and warning if relaxed

---

## Implementation Details

### Weight Normalization

**Automatic:** If weights don't sum to 1.0:

```python
def normalize_weights(alpha, beta, gamma):
    total = alpha + beta + gamma
    if abs(total - 1.0) > 1e-6:
        return alpha/total, beta/total, gamma/total
    return alpha, beta, gamma
```

### Distance Matrix Caching

All distance matrices are computed once and cached in memory:

```python
# Compute once
D_visual = pairwise_distances(features, metric='euclidean')
D_spatial = pairwise_haversine_distances(coords)
D_temporal = pairwise_temporal_distances(years)

# Reuse many times
D_combined = alpha*normalize(D_visual) + beta*normalize(D_spatial) + gamma*normalize(D_temporal)
```

**Memory:** ~800 KB per distance matrix for 673 tiles

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | Selected subset of tiles |
| $\mathcal{D}$ | Full dataset of tiles |
| $\mathbf{f}_i$ | Feature vector for tile $i$ |
| $D(i,j)$ | Distance between tiles $i$ and $j$ |
| $\alpha, \beta, \gamma$ | Weights for visual, spatial, temporal diversity |
| $d_{\min}$ | Minimum spatial distance constraint (km) |
| $k$ | Number of tiles to select |
| $n$ | Total number of tiles in dataset |

---

**Related Documentation:**
- [Architecture Overview](architecture.md) – System design
- [Methodology](methodology.md) – Parameter optimization phases
- [Scientific Background](../08_GOVERNANCE/SCIENTIFIC_BACKGROUND.md) – Mathematical proofs & theory
