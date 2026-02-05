# ⚠️ DEPRECATED: Siehe `concepts.md`

Diese Datei ist **veraltet**. Der Inhalt wurde in [concepts.md](concepts.md) mit vollständiger mathematischer Dokumentation integriert.

**Übersicht der Multi-Criteria Facility Location:**
- Objekt: Maximiere Abdeckung über visuelle, räumliche & zeitliche Distanzen
- Algorithmus: Submodular Greedy Facility Location (apricot-select)
- Gewichtung: $d = \alpha \cdot d_{visual} + \beta \cdot d_{spatial} + \gamma \cdot d_{temporal}$
- Hard-Constraint: Räumlicher Mindestabstand $d_{min}$ mit Fallback-Logik

Siehe [concepts.md](concepts.md) für:
- Vollständige mathematische Formulierung
- Greedy Algorithmus Pseudocode
- Detaillierte Constraint-Behandlung
- Links zu Implementierung in [src/lazy_facility_location.py](../../src/lazy_facility_location.py)

**Status:** Archived (Konsolidiert in 02_THEORY/concepts.md am 2. Februar 2026)
