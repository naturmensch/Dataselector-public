# Archive Procedure (Governance-Evidence Safe)

## Zweck

Diese Prozedur stellt sicher, dass governance- und wissenschaftlich relevante
Evidenz bei Archivierung oder Umstrukturierung reproduzierbar erhalten bleibt.

## Geltungsbereich

1. Alle Dateien in evidenzfuehrenden Dokumentationszonen.
2. Alle Archivwellen unter `docs/07_ARCHIVE/`.
3. Historische Kontexte, die fuer methodische Herleitung referenziert werden.

## Verbindliche Schritte vor Archivierung

1. Scope definieren:
   - Welche Dateien/Ordner werden verschoben oder archiviert?
   - Welcher wissenschaftliche Kontext wird dadurch beruehrt?
2. Snapshot erstellen:
   - Vollstaendige Dateiliste erfassen.
   - SHA256-Pruefsummen fuer alle zu archivierenden Dateien erzeugen.
3. MANIFEST erstellen:
   - Datei `MANIFEST.md` im Ziel-Archivordner anlegen.
   - Muss enthalten:
     - Scope und Datum
     - Zielort der Archivwelle
     - Pruefsummenliste (SHA256)
     - Restore-Prozedur
4. Referenzpfade absichern:
   - Falls aktive Dokumente auf archivierte Inhalte verweisen, diese Verweise
     unmittelbar aktualisieren.
5. Erst danach verschieben/archivieren.

## Restore-Prozedur (Mindeststandard)

1. Archivwelle und `MANIFEST.md` lokalisieren.
2. Dateiintegritaet pruefen:
   - SHA256 fuer wiederherzustellende Dateien berechnen.
   - Mit Manifest-Eintrag vergleichen.
3. Zielzone wiederherstellen:
   - Primary Evidence in aktive Evidenzzone:
     `docs/06_REFERENCE/thesis_decision_evidence/`
   - Secondary Evidence bleibt als Kontext in `docs/07_ARCHIVE/` oder wird als
     explizit gekennzeichnete Kopie bereitgestellt.
4. Protokollreferenzen aktualisieren:
   - Betroffene Status- oder Governance-Dokumente aktualisieren.

## Loesch- und Bereinigungsregel

1. Kein Loeschen von governance- oder evidenzrelevanten Artefakten ohne
   vorgaengiges, versioniertes `MANIFEST.md` mit SHA256.
2. Keine stillen Bereinigungen in evidenzfuehrenden Zonen.

## Quartalsweiser Integrity-Audit

1. Alle `docs/07_ARCHIVE/**/MANIFEST.md` auflisten.
2. Vorhandensein und Versionshistorie pruefen.
3. Stichprobenartige SHA256-Verifikation gegen Manifest durchfuehren.
4. Audit-Ergebnis in Governance-Notiz dokumentieren.

## Verantwortlichkeit

1. Ausfuehrende Person der Archivwelle erstellt und committed `MANIFEST.md`.
2. Review-Person bestaetigt Restore-Prozedur und Integritaetsangaben.
